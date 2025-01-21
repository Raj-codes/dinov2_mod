from dinov2_pipeline.model_initiator import get_dino_finetuned

# User inputs
model_type = 'vits14'  # Choose 'vits14' or 'vitg14'
weights_path = 'C:/Users/rrai/dinov2/pretrained/dinov2_vits_NCT_10k_training_1999_teacher_checkpoint.PTH'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = get_dino_finetuned(model_type, weights_path, device=device)
