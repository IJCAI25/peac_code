This is the implementation for inference of PEAC
described in:

> Perception Emulating Actor-Critic in Crowd Scenario Navigation

## Quick Start
### Prerequisites
- Python 3.x
- pip
- Minimum of 20GB VRAM for running evaluations (We use RTX 3090)
- Minimum of 40GB VRAM for training (We use Tesla A100)

### ⚙ Setup
1. **Install required dependencies**  
Download the dataset and required weight files from [This temporary repository](https://new.space/s/QcbwEh2fGqzt42CY_hx9NQ#HGxKh1cQ94PkY4FeqntAlSy7TVKcjwogTMjoyu4MvDs).
Place the `train_data.pkl` file into the `data` folder, and place the `adapter_model.bin` and `optimizer.pt` files into the `models\weights\lora-alpha_1\checkpoint-11800` folder.

2. **Install required dependencies**  

    ```sh
    pip install -r requirements.txt
    ```

3. **Set up WandB API key**  

    Set up your [WandB](https://wandb.ai/) API key for training and evaluation logging.

    ```sh
    export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```
    
### 🏄 Evaluation

1. **Evaluation for the Decision-Maker**

    Run the following command:

    ```sh
    python train.py \
        --mode eval \
        --resume_from_checkpoint models/weights/lora-alpha_1/checkpoint-11800/ \
        --data_path data/train_data.pkl \
        --val_data_path data/test_data.pkl \
        --eval_items caption,action \
        --vqa
    ```

2. **Evaluation for the Interpreter**

    Run the following command:

    ```sh
    python train.py \
        --mode eval \
        --resume_from_checkpoint models/weights/lora-alpha_1/checkpoint-11800/ \
        --data_path data/train_data.pkl \
        --val_data_path data/test_data.pkl \
        --eval_items vqa \
        --vqa
    ```
