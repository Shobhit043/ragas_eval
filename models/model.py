import os

from sentence_transformers import SentenceTransformer


class EmbeddModel:
    def __init__(self, model_dir_path="checkpoints", model_name="all-MiniLM-L6-v2"):
        CWD = os.path.dirname(os.path.realpath(__file__))
        self.model_checkpoints_path = os.path.join(
            os.path.join(CWD, model_dir_path), "all-MiniLM-L6-v2"
        )

        if not os.path.exists(self.model_checkpoints_path):
            print("Model checkpoints does not exist. Downloading model...")
            model = SentenceTransformer(model_name)
            model.save(self.model_checkpoints_path)
            print(f"Model saved to {self.model_checkpoints_path}")

    def load_model(self):
        return SentenceTransformer(self.model_checkpoints_path)
