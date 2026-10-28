from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent

model_path = BASE_DIR / "student_performance_model.pkl"

model = joblib.load(model_path)

print("=" * 50)
print("SAVED MODEL INFORMATION")
print("=" * 50)

print("\nModel type:")
print(type(model).__name__)

if hasattr(model, "named_steps"):
    print("\nPipeline steps:")

    for step_name, step_object in model.named_steps.items():
        print(f"{step_name}: {type(step_object).__name__}")

    final_model = model.named_steps.get("model")

    if final_model is not None:
        print("\nFinal estimator:")
        print(type(final_model).__name__)

        if hasattr(final_model, "classes_"):
            print("\nClasses:")
            print(final_model.classes_)

print("\nModel inspection completed.")