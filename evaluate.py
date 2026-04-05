from ultralytics import YOLO
import sys

def evaluate_model():
    print("=" * 60)
    print("⚽ EVALUATING YOLO MODEL ON TEST DATASET ⚽")
    print("=" * 60)
    
    try:
        # Load the best model
        print("[INFO] Loading model 'models/best.pt'...")
        model = YOLO('models/best.pt')
        
        # Run validation on the test split
        # We use split='test' to evaluate on unseen data.
        print("[INFO] Running evaluation on test dataset...")
        
        # We pass the absolute path to data.yaml to avoid relative path confusion
        import os
        data_path = os.path.abspath('training/football-players-detection-1/data.yaml')
        
        # In case test path in yaml is broken, we can also override test dir if needed, 
        # but val() uses the paths in data.yaml. Usually test path is fine.
        metrics = model.val(data=data_path, split='test')
        
        # Extract metrics
        # accuracy approximation = precision (True Positives / All predicted Positives)
        # However, typically in object detection we present True Positives/Total Predictions as precision.
        precision = metrics.box.mean_results()[0]  # P
        recall = metrics.box.mean_results()[1]     # R
        map50 = metrics.box.map50                  # mAP@0.5
        map75 = metrics.box.map                    # mAP@0.5:0.95 (Ultralytics defaults 'map' to mean over all thresholds)
        
        # Approximation of generic Accuracy using F1 score or Precision. We will use precision and recall.
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)

        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION METRICS REPORT 📊")
        print("=" * 60)
        
        print("1. Detection Accuracy (Precision / F1 Proxy):")
        print("   - Measures the correctness of object detection.")
        print("   - Formula: TP / (TP + FP)  [For Precision]")
        print(f"   - Precision: {precision:.2%}")
        print(f"   - Recall:    {recall:.2%}")
        print(f"   - F1-Score:  {f1_score:.2%} (Overall Accuracy proxy)")
        
        print("\n2. mAP (Mean Average Precision):")
        print("   - Standard object detection metric across confidence thresholds.")
        print(f"   - mAP@0.5:      {map50:.2%}  (Excellent for basic tracking)")
        print(f"   - mAP@0.5:0.95: {map75:.2%}  (Overall Bounding Box Quality)")
        
        print("\n3. ID Switch Rate (Tracker Metric):")
        print("   - Note: YOLO evaluates Detection, not Tracking.")
        print("   - To get ID Switch Rate, you would run a MOT (Multiple Object Tracking)")
        print("     benchmark sequence which requires bounding boxes linked chronologically.")
        print("   - Without sequential ground truth labels for the track IDs, we rely on mAP.")
        print("     High mAP ensures fewer False Negatives, directly reducing ID Switches!")
        
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    evaluate_model()
