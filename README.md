# Plant Saver API

## Setup

1. **Export your Teachable Machine model**
   - In Teachable Machine, after training: **Export Model** → **Tensorflow** tab (not TF.js) → **Convert model** → download.
   - Unzip it. You'll get `keras_model.h5` and `labels.txt`.
   - Create a `model/` folder in this project and put both files inside:
     ```
     plant_saver_api/
       model/
         keras_model.h5
         labels.txt
     ```

2. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```

3. **Set your Anthropic API key** (for the "dive deeper" explanation)
   ```
   export ANTHROPIC_API_KEY=your_key_here
   ```

4. **Run the server**
   ```
   uvicorn app:app --reload
   ```
   It'll start at `http://127.0.0.1:8000`

## Testing it

With the server running, test with curl:

```bash
curl -X POST http://127.0.0.1:8000/diagnose \
  -F "plant_image=@/path/to/whole_plant.jpg" \
  -F "leaf_image=@/path/to/leaf_closeup.jpg"
```

`leaf_image` is optional — you can omit it and just send `plant_image`.

## Response format

```json
{
  "diagnosis": "Overwatered",
  "confidence": 0.83,
  "all_scores": {
    "Healthy": 0.05,
    "Underwatered": 0.02,
    "Overwatered": 0.83,
    "Pest Damage": 0.06,
    "Sunburn": 0.04
  },
  "used_leaf_photo": true,
  "deep_dive": "... Claude's explanation and care steps ..."
}
```

## Notes

- `labels.txt` from Teachable Machine must list classes in the format `0 Healthy`, `1 Underwatered`, etc. — the loader parses that format automatically.
- If both `plant_image` and `leaf_image` are given, the leaf closeup is weighted more heavily (60/40) since it's usually more diagnostic.
- `deep_dive` can be turned off per-request by sending `deep_dive=false` as a form field, if you just want the raw classification without an API call.
