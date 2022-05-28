import torch
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, OpenAIClipAdapter, \
    DiffusionPriorTrainer, DALLE2

depth = 12
d_model = 768
dprior_path = "./conditioned-prior/vit-b-50k.pth"
clip = OpenAIClipAdapter(clip_choice="ViT-B/32")

# decoder (with unet)

unet1 = Unet(
    dim=128,
    image_embed_dim=512,
    cond_dim=128,
    channels=3,
    dim_mults=(1, 2, 4, 8)
)

unet2 = Unet(
    dim=16,
    image_embed_dim=512,
    cond_dim=128,
    channels=3,
    dim_mults=(1, 2, 4, 8, 16)
)

decoder = Decoder(
    unet=(unet1, unet2),
    image_sizes=(128, 256),
    clip=clip,
    timesteps=100,
    image_cond_drop_prob=0.1,
    text_cond_drop_prob=0.5,
    condition_on_text_encodings=False  # set this to True if you wish to condition on text during training and sampling
)


def load_diffusion_model(dprior_path, device=None, clip_choice="ViT-B/32"):
    loaded_obj = torch.load(str(dprior_path), map_location='cpu')

    if clip_choice == "ViT-B/32":
        dim = 512
    else:
        dim = 768

    prior_network = DiffusionPriorNetwork(
        dim=dim,
        depth=12,
        dim_head=64,
        heads=12,
        normformer=True
    )

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter(clip_choice),
        image_embed_dim=dim,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
    )

    diffusion_prior.load_state_dict(loaded_obj["model"], strict=True)

    diffusion_prior = DiffusionPriorTrainer(
        diffusion_prior=diffusion_prior,
        lr=1.1e-4,
        wd=6.02e-2,
        max_grad_norm=0.5,
        amp=False,
    )

    diffusion_prior.optimizer.load_state_dict(loaded_obj['optimizer'])
    diffusion_prior.scaler.load_state_dict(loaded_obj['scaler'])

    return diffusion_prior


diffusion_prior = load_diffusion_model(dprior_path)

# do above for many steps

dalle2 = DALLE2(
    prior=diffusion_prior,
    decoder=decoder
)

images = dalle2(
    ['a butterfly trying to escape a tornado'],
    cond_scale=2.  # classifier free guidance strength (> 1 would strengthen the condition)
)
