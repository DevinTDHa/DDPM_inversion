from typing import Optional
import torch
import os
from tqdm import tqdm
from ddm_inversion.models import decode_latents
from ddm_inversion.utils import (
    pil_to_tensor,
    save_intermediate_img,
    project_x_to_normal_space,
)
from prompt_to_prompt.ptp_classes import AttentionStore
import pdb


def load_real_image(folder="data/", img_name=None, idx=0, img_size=512, device="cuda"):
    from PIL import Image
    from glob import glob

    if img_name is not None:
        path = os.path.join(folder, img_name)
    else:
        path = glob(folder + "*")[idx]

    img = Image.open(path).resize((img_size, img_size))

    img = pil_to_tensor(img).to(device)

    if img.shape[1] == 4:
        img = img[:, :3, :, :]
    return img


def mu_tilde(model, xt, x0, timestep):
    "mu_tilde(x_t, x_0) DDPM paper eq. 7"
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps
    )
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    alpha_t = model.scheduler.alphas[timestep]
    beta_t = 1 - alpha_t
    alpha_bar = model.scheduler.alphas_cumprod[timestep]
    return ((alpha_prod_t_prev**0.5 * beta_t) / (1 - alpha_bar)) * x0 + (
        (alpha_t**0.5 * (1 - alpha_prod_t_prev)) / (1 - alpha_bar)
    ) * xt


def sample_xts_from_x0(model, x0, num_inference_steps=50):
    """
    Samples from P(x_1:T|x_0)
    """
    # torch.manual_seed(43256465436)
    alpha_bar = model.scheduler.alphas_cumprod
    sqrt_one_minus_alpha_bar = (1 - alpha_bar) ** 0.5
    alphas = model.scheduler.alphas
    betas = 1 - alphas
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size,
    )

    timesteps = model.scheduler.timesteps.to(model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xts = torch.zeros(
        (
            num_inference_steps + 1,
            model.unet.in_channels,
            model.unet.sample_size,
            model.unet.sample_size,
        )
    ).to(x0.device)
    xts[0] = x0
    for t in reversed(timesteps):
        idx = num_inference_steps - t_to_idx[int(t)]
        xts[idx] = (
            x0 * (alpha_bar[t] ** 0.5)
            + torch.randn_like(x0) * sqrt_one_minus_alpha_bar[t]
        )

    return xts


def encode_text(model, prompts):
    text_input = model.tokenizer(
        prompts,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        text_encoding = model.text_encoder(text_input.input_ids.to(model.device))[0]
    return text_encoding


def forward_step(model, model_output, timestep, sample):
    next_timestep = min(
        model.scheduler.config.num_train_timesteps - 2,
        timestep
        + model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps,
    )

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    # alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep] if next_ltimestep >= 0 else self.scheduler.final_alpha_cumprod

    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)

    # 5. TODO: simple noising implementatiom
    next_sample = model.scheduler.add_noise(
        pred_original_sample, model_output, torch.LongTensor([next_timestep])
    )
    return next_sample


def get_variance(model, timestep):  # , prev_timestep):
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps
    )
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    return variance


def inversion_forward_process(
    model,
    x0,
    etas=1,
    prog_bar=False,
    prompt="",
    cfg_scale=3.5,
    num_inference_steps=50,
    eps=None,
):
    if not prompt == "":
        text_embeddings = encode_text(model, prompt)
    uncond_embedding = encode_text(model, "")
    timesteps = model.scheduler.timesteps.to(model.device)
    variance_noise_shape = (
        num_inference_steps,
        model.unet.in_channels,
        model.unet.sample_size,
        model.unet.sample_size,
    )
    if etas is None or (type(etas) in [int, float] and etas == 0):
        eta_is_zero = True
        zs = None
    else:
        eta_is_zero = False
        if type(etas) in [int, float]:
            etas = [etas] * model.scheduler.num_inference_steps
        xts = sample_xts_from_x0(model, x0, num_inference_steps=num_inference_steps)
        alpha_bar = model.scheduler.alphas_cumprod
        zs = torch.zeros(size=variance_noise_shape, device=model.device)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0
    # op = tqdm(reversed(timesteps)) if prog_bar else reversed(timesteps)
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        # idx = t_to_idx[int(t)]
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        # 1. predict noise residual
        if not eta_is_zero:
            xt = xts[idx + 1][None]
            # xt = xts_cycle[idx+1][None]

        with torch.no_grad():
            out = model.unet.forward(
                xt, timestep=t, encoder_hidden_states=uncond_embedding
            )
            if not prompt == "":
                cond_out = model.unet.forward(
                    xt, timestep=t, encoder_hidden_states=text_embeddings
                )

        if not prompt == "":
            ## classifier free guidance
            noise_pred = out.sample + cfg_scale * (cond_out.sample - out.sample)
        else:
            noise_pred = out.sample

        if eta_is_zero:
            # 2. compute more noisy image and set x_t -> x_t+1
            xt = forward_step(model, noise_pred, t, xt)
        else:
            # xtm1 =  xts[idx+1][None]
            xtm1 = xts[idx][None]
            # pred of x0
            pred_original_sample = (
                xt - (1 - alpha_bar[t]) ** 0.5 * noise_pred
            ) / alpha_bar[t] ** 0.5

            # direction to xt
            prev_timestep = (
                t
                - model.scheduler.config.num_train_timesteps
                // model.scheduler.num_inference_steps
            )
            alpha_prod_t_prev = (
                model.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else model.scheduler.final_alpha_cumprod
            )

            variance = get_variance(model, t)
            pred_sample_direction = (1 - alpha_prod_t_prev - etas[idx] * variance) ** (
                0.5
            ) * noise_pred

            mu_xt = (
                alpha_prod_t_prev ** (0.5) * pred_original_sample
                + pred_sample_direction
            )

            z = (xtm1 - mu_xt) / (etas[idx] * variance**0.5)
            zs[idx] = z

            # correction to avoid error accumulation
            xtm1 = mu_xt + (etas[idx] * variance**0.5) * z
            xts[idx] = xtm1

    if not zs is None:
        zs[0] = torch.zeros_like(zs[0])

    return xt, zs, xts


def reverse_step(model, model_output, timestep, sample, eta=0, variance_noise=None):
    """
    Perform a reverse step in the denoising diffusion probabilistic model (DDPM) process.
    Args:
        model (nn.Module): The model used for the DDPM process.
        model_output (torch.Tensor): The output from the model at the current timestep.
        timestep (int): The current timestep in the diffusion process.
        sample (torch.Tensor): The current sample in the diffusion process.
        eta (float, optional): The scaling factor η for the variance noise. Default is 0.
        variance_noise (torch.Tensor, optional): The noise to be added to the sample. If None, random noise is generated. Default is None.
    Returns:
        torch.Tensor: The sample at the previous timestep in the diffusion process.
    """
    # 1. get previous step value (=t-1)
    prev_timestep = (
        timestep
        - model.scheduler.config.num_train_timesteps
        // model.scheduler.num_inference_steps
    )

    # 2. compute alphas, betas
    alpha_prod_t = model.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        model.scheduler.alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else model.scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t

    # 3. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf (DDIM)
    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)

    # 5. compute variance: "sigma_t(η)" -> see formula (16)
    # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
    # variance = self.scheduler._get_variance(timestep, prev_timestep)
    variance = get_variance(model, timestep)  # , prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    # Take care of asymetric reverse process (asyrp)
    model_output_direction = model_output

    # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf (DDIM)
    # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * model_output_direction
    pred_sample_direction = (1 - alpha_prod_t_prev - eta * variance) ** (
        0.5
    ) * model_output_direction

    # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    # 8. Add noise if eta > 0, DHA: DDPM Case if eta == 1
    if eta > 0:
        if variance_noise is None:
            variance_noise = torch.randn(model_output.shape, device=model.device)
        sigma_z = eta * variance ** (0.5) * variance_noise
        prev_sample = prev_sample + sigma_z

    return prev_sample


def inversion_reverse_process(
    model,
    xT,
    etas=1,
    prompts="",
    cfg_scales=None,
    prog_bar=False,
    zs=None,
    controller=None,
    asyrp=False,
):
    """
    Perform the reverse process of inversion using a given model.

    Args:
        model (torch.nn.Module): The model to use for the reverse process.
        xT (torch.Tensor): The initial tensor to start the reverse process.
        etas (float): The scaling factor η for the variance noise. Default is 1.
        prompts (str or list of str, optional): Text prompts for conditional generation. Default is an empty string.
        cfg_scales (list of floats, optional): Classifier-free guidance scales. Default is None.
        prog_bar (bool, optional): Whether to display a progress bar. Default is False.
        zs (torch.Tensor, optional): Latent variables for each timestep. Default is None.
        controller (object, optional): An optional controller object with a step_callback method. Default is None.
        asyrp (bool, optional): An optional flag for asynchronous reverse process. Default is False.
    Returns:
        tuple: A tuple containing the final tensor after the reverse process and the latent variables (zs).
    """
    batch_size = len(prompts)

    cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device)

    text_embeddings = encode_text(model, prompts)
    uncond_embedding = encode_text(model, [""] * batch_size)

    if etas is None:
        etas = 0
    if type(etas) in [int, float]:
        etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    xt = xT.expand(batch_size, -1, -1, -1)
    op = tqdm(timesteps[-zs.shape[0] :]) if prog_bar else timesteps[-zs.shape[0] :]

    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0] :])}

    for t in op:
        idx = (
            model.scheduler.num_inference_steps
            - t_to_idx[int(t)]
            - (model.scheduler.num_inference_steps - zs.shape[0] + 1)
        )
        ## Unconditional embedding
        with torch.no_grad():
            uncond_out = model.unet.forward(
                xt, timestep=t, encoder_hidden_states=uncond_embedding
            )

        ## Conditional embedding
        if prompts:
            with torch.no_grad():
                cond_out = model.unet.forward(
                    xt, timestep=t, encoder_hidden_states=text_embeddings
                )

        z = zs[idx] if not zs is None else None
        z = z.expand(batch_size, -1, -1, -1)
        if prompts:
            ## classifier free guidance
            noise_pred = uncond_out.sample + cfg_scales_tensor * (
                cond_out.sample - uncond_out.sample
            )
        else:
            noise_pred = uncond_out.sample
        # 2. compute less noisy image and set x_t -> x_t-1
        xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)
        if controller is not None:
            xt = controller.step_callback(xt)  # DHA: By default just identity?
    return xt, zs


def regr_loss_fn(x, x_counterf, y, y_hat, lambd=1.0):
    """
    Compute the regression loss function. Uses the formulation by (Wachter 2017).
    """
    return lambd * torch.nn.functional.mse_loss(
        y, y_hat
    ) + torch.nn.functional.mse_loss(x, x_counterf)


def inversion_reverse_process_grad_guided(
    model,
    x0: torch.Tensor,
    xT: torch.Tensor,
    predictor: torch.nn.Module,
    zs: torch.Tensor,
    etas: int = 0,
    cfg_scales=None,
    controller: Optional[AttentionStore] = None,
    time_stamp: str = "",
):
    assert predictor, "A predictor module is required for grad guided inversion"

    batch_size = 1

    # cfg_scales_tensor = torch.Tensor(cfg_scales).view(-1, 1, 1, 1).to(model.device)
    # text_embeddings = None  # no text embeddings needed

    # 1. Create Unconditional embeddings
    uncond_embedding = encode_text(model, [""] * batch_size).requires_grad_(False)

    if etas is None:
        etas = 0
    if type(etas) in [int, float]:
        etas = [etas] * model.scheduler.num_inference_steps
    assert len(etas) == model.scheduler.num_inference_steps
    timesteps = model.scheduler.timesteps.to(model.device)

    def init_xt():
        return xT.clone().expand(batch_size, -1, -1, -1)

    # DHA: might need to adjust this if we don't have enough memory to store all grads
    xt = init_xt()
    # 2. Get required timesteps
    t_to_idx = {int(v): k for k, v in enumerate(timesteps[-zs.shape[0] :])}

    grads_for = "zs"
    grad_params = []
    # Enable Grads for z
    # DHA: maybe do this iteratively in the loop? keep denoising and adjusting zs until the predictor value changes?
    zs_orig = zs.detach().clone()
    zs.requires_grad_()  # DHA: where do we need to start to collect grads?
    grad_params.append(zs)

    optimizer = torch.optim.Adam(grad_params, lr=0.005)

    xt_decoded = decode_latents(model, xt)
    pred = predictor(project_x_to_normal_space(xt_decoded))

    print("Initial predictor value before denoising: ", pred.item())
    intermediate_folder = f"imgs/intermediate/reg{time_stamp}_" + str(pred.item())
    os.makedirs(intermediate_folder, exist_ok=True)  # DHA: For intermediate images

    reg_n = 0
    reg_n_max = 10
    threshold = 0.9  # TODO
    target = 1.0
    y = torch.Tensor([target]).to(x0.device)

    while pred < threshold and reg_n < reg_n_max:
        # Reset xt
        xt = init_xt()

        # 3. Start reverse process loop
        for t in tqdm(timesteps[-zs.shape[0] :], desc=f"Regression run {reg_n}"):
            idx = (
                model.scheduler.num_inference_steps
                - t_to_idx[int(t)]
                - (model.scheduler.num_inference_steps - zs.shape[0] + 1)
            )

            # 3.1 Predict noise
            with torch.no_grad():
                uncond_out = model.unet.forward(
                    xt, timestep=t, encoder_hidden_states=uncond_embedding
                )

            # ## Conditional embedding
            # if prompts:
            #     with torch.no_grad():
            #         cond_out = model.unet.forward(
            #             xt, timestep=t, encoder_hidden_states=text_embeddings
            #         )

            z = zs[idx]
            z = z.expand(batch_size, -1, -1, -1)

            # if prompts:
            #     ## classifier free guidance
            #     noise_pred = uncond_out.sample + cfg_scales_tensor * (
            #         cond_out.sample - uncond_out.sample
            #     )
            # else:

            noise_pred = uncond_out.sample
            # 3.2 compute less noisy image and set x_t -> x_t-1
            xt = reverse_step(model, noise_pred, t, xt, eta=etas[idx], variance_noise=z)
            if controller is not None:
                xt = controller.step_callback(xt)  # DHA: By default just identity?

        # Prepare Gradients
        xt_decoded = decode_latents(model, xt, mixed_precision=False)
        xt_decoded.requires_grad_()
        pred: torch.Tensor = predictor(project_x_to_normal_space(xt_decoded))
        print("Predictor value after denoising: ", pred.item())
        save_intermediate_img(
            intermediate_folder + f"/reg_{reg_n}_{pred.item():.4f}.png",
            xt_decoded,
        )
        loss = regr_loss_fn(x0, xt_decoded, y, pred)
        loss.backward(inputs=grad_params)

        # 4. Optimize zs
        print(f"t: {t.item()}; Grad Magnitudes: ", zs.grad.abs().sum().item())
        optimizer.step()
        optimizer.zero_grad()

        reg_n += 1

    # Edit
    return xt, z
