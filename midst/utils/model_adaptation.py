from midst.models.midst import MIDST

import torch


def load_trained_model(
        model_ckpt_path: str,
        trained_model_params: dict,
        new_model_params: dict,
        model_type: type = MIDST,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        load_partial_dynamics: bool = True,
        load_encoder_decoder: bool = True,
        fine_tune: bool = False,
):
    # Instantiate the new model
    new_model = model_type(
        **new_model_params
    )

    # Load the trained model
    ckpt = torch.load(model_ckpt_path, map_location=device)['model']
    trained_model = model_type(
        **trained_model_params
    )
    trained_model.load_state_dict(ckpt, strict=False,)
    trained_model.to(device)

    if load_encoder_decoder:
        # Copy the existing systems and freeze their parameters
        new_model.encoder = trained_model.encoder
        new_model.decoder = trained_model.decoder

        if not fine_tune:
            new_model.encoder.requires_grad_(False)
            new_model.decoder.requires_grad_(False)

    if load_partial_dynamics:
        new_model._U_per_t = trained_model._U_per_t
        new_model._V_per_t = trained_model._V_per_t

        new_model._U_per_t.requires_grad_(False)
        new_model._V_per_t.requires_grad_(False)

    # Return the new model, with only the new system as trainable parameter
    return new_model
