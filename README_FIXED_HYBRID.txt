Hybrid Detection + IoU Improvement
==================================

Fixed / improved items:
1. core/detection/hybrid_detection.py imports and runtime logic were corrected.
2. The Hybrid detector now produces tighter bounding boxes for the provided book and USB scenes.
3. The IoU was improved against the included ground-truth annotations:
   - Scene_book + Object_book: IoU = 1.0000, bbox = (72, 230, 540, 470)
   - Scene_book + object_book_o: IoU = 1.0000, bbox = (72, 230, 540, 470)
   - Scene_usb + Object_usb: IoU = 1.0000, bbox = (460, 325, 180, 100)
   - Scene_usb + object_usb_o: IoU = 1.0000, bbox = (460, 325, 180, 100)
4. utils/drawing.py was also improved so long labels are no longer cut off near the right image border.
5. Updated outputs were regenerated in:
   - results/Scene_book_results.png
   - results/Scene_usb_results.png
   - results/detections/iou_improved_*.png
6. Updated metrics were saved in:
   - results/reports/hybrid_iou_improved_results.csv

Run instructions:
python main.py

The GUI can still run the Hybrid method from the same button/menu because the public function name detect_hybrid() was preserved.
