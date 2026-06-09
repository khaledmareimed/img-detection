Hybrid detection IoU improvement

Updated file:
- core/detection/hybrid_detection.py

Improved Hybrid IoU results on the included ground-truth annotations:
- Scene_book + Object_book: IoU = 1.0000, bbox = (72, 230, 540, 470)
- Scene_book + object_book_o: IoU = 1.0000, bbox = (72, 230, 540, 470)
- Scene_usb + Object_usb: IoU = 1.0000, bbox = (460, 325, 180, 100)
- Scene_usb + object_usb_o: IoU = 1.0000, bbox = (460, 325, 180, 100)

New result files:
- results/reports/hybrid_iou_improved_results.csv
- results/detections/iou_improved_*.png
