import cv2
import numpy as np
import os # For checking if reference image exists

# --- Configuration ---
REFERENCE_IMAGE_PATH = 'reference_cylinder.jpg' # <<== CREATE THIS IMAGE FIRST!
MIN_MATCH_COUNT = 15  # Minimum number of good matches to consider a detection
FLANN_INDEX_KDTREE = 1 # Not used for ORB but good to know for SIFT/SURF
LOWES_RATIO_TEST_THRESHOLD = 0.7 # For SIFT/SURF, for ORB we often just sort

def detect_object_orb(frame, ref_img_gray, ref_kps, ref_des, orb, bf_matcher):
    """
    Detects the reference object in the current frame using ORB.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors for the current frame
    try:
        kps_frame, des_frame = orb.detectAndCompute(frame_gray, None)
    except cv2.error as e:
        print(f"Error in ORB detectAndCompute for frame: {e}")
        return frame, None # Return original frame if error

    if des_frame is None or len(des_frame) < 2: # Need at least 2 for KNN match
        return frame, None

    # Match descriptors using Brute-Force Matcher
    # For ORB, Hamming distance is appropriate. k=2 means find 2 nearest neighbors
    matches = bf_matcher.knnMatch(ref_des, des_frame, k=2)

    # Apply Lowe's ratio test (or simply filter good matches)
    good_matches = []
    if matches:
        for m_pair in matches:
            if len(m_pair) == 2: # Ensure we got two matches
                m, n = m_pair
                if m.distance < LOWES_RATIO_TEST_THRESHOLD * n.distance: # Lowe's ratio
                    good_matches.append(m)
            elif len(m_pair) == 1: # Sometimes only one match is found per query descriptor
                # For ORB, sometimes just taking matches below a certain distance threshold works
                # or sorting and taking top N. Here we'll stick to ratio test analogy.
                # If only one match, we can't do ratio test. Could add it if distance is very small.
                pass


    # --- Visualization ---
    # Draw all matches (can be noisy, useful for debugging)
    # img_matches_debug = cv2.drawMatchesKnn(ref_img_gray, ref_kps, frame_gray, kps_frame, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("All Matches Debug", img_matches_debug)

    if len(good_matches) > MIN_MATCH_COUNT:
        # Extract location of good matches
        src_pts = np.float32([ref_kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography
        try:
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print("Homography not found.")
                return frame, None
        except cv2.error as e:
            print(f"Error in findHomography: {e}")
            return frame, None


        matchesMask = mask.ravel().tolist()

        # Get dimensions of the reference image
        h, w = ref_img_gray.shape
        # Define corners of the reference image
        pts_ref_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # Transform reference corners to frame coordinates
        if M is not None:
            dst_corners = cv2.perspectiveTransform(pts_ref_corners, M)
            # Draw a polygon around the detected object in the frame
            frame_with_detection = cv2.polylines(frame, [np.int32(dst_corners)], True, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(frame_with_detection, "Cylinder Detected (ORB)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

            # Draw only good matches
            img_good_matches = cv2.drawMatches(ref_img_gray, ref_kps, frame_with_detection, kps_frame,
                                              good_matches, None, # Pass frame_with_detection here
                                              matchesMask=matchesMask, # draw only inliers
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            return img_good_matches, dst_corners
        else:
            return frame, None # Homography failed

    else:
        # print(f"Not enough good matches found - {len(good_matches)}/{MIN_MATCH_COUNT}")
        matchesMask = None
        # Draw all good matches even if not enough for homography (for debugging)
        # img_good_matches = cv2.drawMatches(ref_img_gray, ref_kps, frame, kps_frame, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # return img_good_matches, None
        return frame, None


if __name__ == '__main__':
    # --- 1. Load Reference Image and Precompute Features ---
    if not os.path.exists(REFERENCE_IMAGE_PATH):
        print(f"ERROR: Reference image not found at '{REFERENCE_IMAGE_PATH}'")
        print("Please create a well-lit image of your brown cylinder and save it as reference_cylinder.jpg")
        exit()

    ref_img_bgr = cv2.imread(REFERENCE_IMAGE_PATH)
    if ref_img_bgr is None:
        print(f"ERROR: Could not load reference image from '{REFERENCE_IMAGE_PATH}'")
        exit()
    ref_img_gray = cv2.cvtColor(ref_img_bgr, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    # You can experiment with nfeatures, scaleFactor, nlevels etc.
    orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE) # Increased features

    # Find keypoints and descriptors for the reference image
    try:
        ref_kps, ref_des = orb.detectAndCompute(ref_img_gray, None)
        if ref_des is None:
            print(f"ERROR: No descriptors found in reference image '{REFERENCE_IMAGE_PATH}'. Is it too plain?")
            exit()
        print(f"Found {len(ref_kps)} keypoints in reference image.")
    except cv2.error as e:
        print(f"Error in ORB detectAndCompute for reference image: {e}")
        exit()


    # Initialize Brute-Force Matcher
    # For ORB (binary descriptor), use NORM_HAMMING.
    # crossCheck=False allows knnMatch. If True, knnMatch will error.
    bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


    # --- 2. Setup Camera ---
    cap = cv2.VideoCapture(0) # 0 for default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("\nStarting camera feed. Press 'q' to quit.")
    print("Feature-based matching can struggle with plain, texture-less objects.")
    print("If detection is poor, your cylinder might lack distinct features for ORB.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        processed_frame, detected_outline = detect_object_orb(frame.copy(), ref_img_gray, ref_kps, ref_des, orb, bf_matcher)

        if detected_outline is not None:
            # You could do something with detected_outline here (e.g., get centroid)
            pass

        cv2.imshow('ORB Object Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
