import cv2

cap = cv2.VideoCapture(0)  # 0 = default webcam
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Haar cascades for face and smile detection (comes with opencv-python)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    

    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    motion_detected = False
    shadow_detected = False
    smile_detected = False
    min_x, min_y, max_x, max_y = None, None, None, None

    fg_mask = back_sub.apply(frame_resized)

    # Clean up mask
    _, thresh = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Shadow detection: in MOG2, shadows are usually mid-level values (around 127)
    shadow_mask = cv2.inRange(fg_mask, 1, 200)  # exclude pure background (0) and strong foreground (255)
    if cv2.countNonZero(shadow_mask) > 1000:
        shadow_detected = True

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # ignore small blobs; lower if you want more sensitivity
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if min_x is None:
            min_x, min_y = x, y
            max_x, max_y = x + w, y + h
        else:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        motion_detected = True

    # Draw a single box that covers all motion
    if motion_detected and min_x is not None:
        cv2.rectangle(frame_resized, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

    # Face + smile detection (on full frame)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)
        if len(smiles) > 0:
            smile_detected = True
            break

    # Status text in top-left corner
    if motion_detected:
        status_text = "Moving object detected"
        color = (0, 0, 255)  # red
    else:
        status_text = "Normal"
        color = (0, 255, 0)  # green

    shadow_text = "Shadow detected" if shadow_detected else "No shadow"
    smile_text = "Smiling" if smile_detected else "Not smiling"

    cv2.putText(
        frame_resized,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_resized,
        shadow_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_resized,
        smile_text,
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Moving Object Detection", frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
