import os
import json

class PresetManager:
    def __init__(self, presets_dir: str):
        self.presets_dir = presets_dir
        os.makedirs(self.presets_dir, exist_ok=True)
        self._generate_default_presets()

    def list_presets(self) -> list[str]:
        """Lists all preset names available in the presets directory."""
        files = [f for f in os.listdir(self.presets_dir) if f.endswith(".json")]
        return sorted([os.path.splitext(f)[0] for f in files])

    def load_preset(self, name: str) -> dict:
        """Loads a preset by name."""
        path = os.path.join(self.presets_dir, f"{name}.json")
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading preset {name}: {e}")
            return {}

    def save_preset(self, name: str, params: dict) -> bool:
        """Saves a preset with the given name and parameters."""
        path = os.path.join(self.presets_dir, f"{name}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving preset {name}: {e}")
            return False

    def _generate_default_presets(self):
        """Generates standard useful presets if the folder is empty."""
        default_preset = {
            "exposure": 0.0,
            "contrast": 0.0,
            "highlights": 0.0,
            "shadows": 0.0,
            "saturation": 0.0,
            "temperature": 5500.0,
            "tint": 0.0,
            "color_noise_reduction": 20.0,
            "lut": "None",
            "curves": {
                "rgb": [[0.0, 0.0], [1.0, 1.0]],
                "red": [[0.0, 0.0], [1.0, 1.0]],
                "green": [[0.0, 0.0], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [1.0, 1.0]]
            }
        }
        self.save_preset("01 - Default Reset", default_preset)

        vibrant_preset = default_preset.copy()
        vibrant_preset.update({
            "exposure": 0.15,
            "contrast": 0.1,
            "saturation": 0.15,
            "highlights": -0.1,
            "shadows": 0.15,
            "temperature": 5700.0,
            "tint": 2.0
        })
        self.save_preset("02 - Vibrant Portrait", vibrant_preset)

        cinematic_preset = default_preset.copy()
        cinematic_preset.update({
            "exposure": -0.05,
            "contrast": 0.15,
            "saturation": 0.05,
            "highlights": -0.15,
            "shadows": 0.1,
            "lut": "Cinematic Teal & Orange.cube"
        })
        self.save_preset("03 - Cinematic Teal & Orange", cinematic_preset)

        mono_preset = default_preset.copy()
        mono_preset.update({
            "exposure": 0.0,
            "contrast": 0.3,
            "highlights": -0.2,
            "shadows": 0.1,
            "saturation": -1.0,  # Pure black and white
            "curves": {
                "rgb": [[0.0, 0.0], [0.25, 0.1], [0.75, 0.9], [1.0, 1.0]],
                "red": [[0.0, 0.0], [1.0, 1.0]],
                "green": [[0.0, 0.0], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [1.0, 1.0]]
            }
        })
        self.save_preset("04 - High Contrast Mono", mono_preset)

        vintage_preset = default_preset.copy()
        vintage_preset.update({
            "exposure": 0.1,
            "contrast": -0.1,
            "saturation": -0.1,
            "highlights": -0.15,
            "shadows": 0.2,
            "lut": "Warm Vintage.cube"
        })
        self.save_preset("05 - Warm Vintage Matte", vintage_preset)
