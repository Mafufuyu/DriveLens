// DriveLens.cpp : Edge camera agent – captures frames and uploads to cloud.
//
// Usage:
//   DriveLens.exe              -> open default webcam (device 0)
//   DriveLens.exe video.mp4    -> read from a video file

#include "DriveLens.h"

// ── Debug switch ─────────────────────────────────────────────────────
// Uncomment the line below to save captured frames to disk and show a
// preview window. Comment it back out for production / release builds.
// #define DEBUG_SAVE_FRAMES

// ── Configuration ────────────────────────────────────────────────────
constexpr const char* API_ENDPOINT        = "http://localhost:8000/upload"; // Debug only
constexpr int         CAPTURE_INTERVAL_SEC = 2;
constexpr int         RESIZE_WIDTH         = 640;
constexpr int         RESIZE_HEIGHT        = 480;
constexpr int         JPEG_QUALITY         = 80;   // 0-100, lower = smaller file

#ifdef DEBUG_SAVE_FRAMES
constexpr const char* DEBUG_OUTPUT_DIR = "debug_frames";
#endif

// ── Helper: encode a cv::Mat frame to an in-memory JPEG buffer ──────
static bool encodeFrameToJpeg(const cv::Mat& frame,
							  std::vector<uchar>& buffer)
{
	std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY };
	return cv::imencode(".jpg", frame, buffer, params);
}

#ifdef DEBUG_SAVE_FRAMES
// ── Helper: save frame to disk & show preview window ─────────────────
static void debugSaveFrame(const cv::Mat& frame, int frameIndex)
{
	// Create output directory on first call
	std::filesystem::create_directories(DEBUG_OUTPUT_DIR);

	std::string path = std::string(DEBUG_OUTPUT_DIR) + "/frame_"
					 + std::to_string(frameIndex) + ".jpg";
	cv::imwrite(path, frame);
	std::cout << "[Debug] Saved " << path << std::endl;

	// Show a live preview window (press any key in window to continue)
	cv::imshow("DriveLens Debug", frame);
	cv::waitKey(1);  // 1 ms – keeps the window responsive
}
#endif

// ── Helper: POST a JPEG buffer to the cloud API ─────────────────────
static bool uploadFrame(const std::vector<uchar>& jpegBuffer,
						int frameIndex)
{
	std::string filename = "frame_" + std::to_string(frameIndex) + ".jpg";

	// Build a multipart/form-data request with the raw JPEG bytes
	std::string body(jpegBuffer.begin(), jpegBuffer.end());
	cpr::Response res = cpr::Post(
		cpr::Url{ API_ENDPOINT },
		cpr::Multipart{
			{ "file", cpr::Buffer{ body.begin(), body.end(), filename } }
		},
		cpr::Timeout{ 10000 }  // 10-second timeout
	);

	if (res.status_code == 200) {
		std::cout << "[Upload] Frame " << frameIndex
				  << " OK  (" << jpegBuffer.size() << " bytes)" << std::endl;
		return true;
	}

	std::cerr << "[Upload] Frame " << frameIndex
			  << " FAILED  status=" << res.status_code
			  << "  error=" << res.error.message << std::endl;
	return false;
}

// ── Main ─────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
	try {
		// --- 1. Open video source (webcam or file) ---
		cv::VideoCapture cap;
		bool isVideoFile = false;

		if (argc >= 2) {
			// Argument provided -> treat as video file path
			std::string videoPath = argv[1];
			cap.open(videoPath);
			isVideoFile = true;
			std::cout << "[DriveLens] Opening video file: " << videoPath << std::endl;
		} else {
			// No argument -> open default webcam
			cap.open(0);
			std::cout << "[DriveLens] Opening webcam (device 0)" << std::endl;
		}

		if (!cap.isOpened()) {
			std::cerr << "[Error] Cannot open video source." << std::endl;
			return 1;
		}

		double fps = cap.get(cv::CAP_PROP_FPS);
		if (fps <= 0) fps = 30.0;   // fallback for webcam

		// Number of frames to skip between captures (≈ every 2 seconds)
		int frameSkip = static_cast<int>(fps * CAPTURE_INTERVAL_SEC);

		std::cout << "[DriveLens] Source FPS: " << fps
				  << "  |  Capture every " << frameSkip << " frames ("
				  << CAPTURE_INTERVAL_SEC << "s)" << std::endl;

		// --- 2. Main capture loop ---
		cv::Mat frame, resized;
		std::vector<uchar> jpegBuffer;
		int frameCount  = 0;
		int captureIndex = 0;

		while (true) {
			if (!cap.read(frame) || frame.empty()) {
				if (isVideoFile) {
					std::cout << "[DriveLens] End of video file." << std::endl;
				} else {
					std::cerr << "[Error] Failed to read frame from webcam." << std::endl;
				}
				break;
			}

			++frameCount;

			// Only process every N-th frame
			if (frameCount % frameSkip != 0) {
				continue;
			}

			// --- 3. Resize for bandwidth savings ---
			cv::resize(frame, resized, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));

			// --- 4. Encode to JPEG ---
			if (!encodeFrameToJpeg(resized, jpegBuffer)) {
				std::cerr << "[Error] JPEG encoding failed for frame "
						  << captureIndex << std::endl;
				continue;
			}

			#ifdef DEBUG_SAVE_FRAMES
			// --- 4.5 Debug: save to disk & show preview ---
			debugSaveFrame(resized, captureIndex);
#endif

			// --- 5. Upload to cloud API ---
			uploadFrame(jpegBuffer, captureIndex);
			++captureIndex;
		}

		cap.release();
#ifdef DEBUG_SAVE_FRAMES
		cv::destroyAllWindows();
#endif
		std::cout << "[DriveLens] Done. Uploaded " << captureIndex
				  << " frames." << std::endl;

	} catch (const std::exception& ex) {
		std::cerr << "[Fatal] " << ex.what() << std::endl;
		return 1;
	}

	return 0;
}
