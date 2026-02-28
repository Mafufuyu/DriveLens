// DriveLens.cpp : Edge camera agent – captures frames, uploads to cloud,
//                  and displays AI detection results with bounding boxes.
//
// Strategy: Cloud-Heavy – object detection runs on the server (YOLOv8).
// The edge device captures, uploads, parses results, and draws overlays.
//
// Usage:
//   DriveLens.exe              -> open default webcam (device 0)
//   DriveLens.exe video.mp4    -> read from a video file

#include "DriveLens.h"

using json = nlohmann::json;

// ── Debug switch ─────────────────────────────────────────────────────
#define DEBUG_SAVE_FRAMES

// ── Configuration ────────────────────────────────────────────────────
constexpr const char* API_ENDPOINT = "http://localhost:8000/upload"; // Change to your server URL
constexpr int         CAPTURE_INTERVAL_SEC = 2;
constexpr int         RESIZE_WIDTH         = 640;
constexpr int         RESIZE_HEIGHT        = 480;
constexpr int         JPEG_QUALITY         = 80;

#ifdef DEBUG_SAVE_FRAMES
constexpr const char* DEBUG_OUTPUT_DIR = "debug_frames";
#endif

// ── Detection data parsed from the server JSON response ──────────────
struct Detection {
	std::string name;
	double      confidence;
	int         x_min, y_min, x_max, y_max;
};

struct CloudResult {
	std::vector<Detection> objects;
	int                    imageWidth  = RESIZE_WIDTH;
	int                    imageHeight = RESIZE_HEIGHT;
};

// ── parseCloudResponse ───────────────────────────────────────────────
static CloudResult parseCloudResponse(const std::string& jsonStr)
{
	CloudResult result;
	if (jsonStr.empty()) return result;

	try {
		auto j = json::parse(jsonStr);

		result.imageWidth  = j.value("image_width",  RESIZE_WIDTH);
		result.imageHeight = j.value("image_height", RESIZE_HEIGHT);

		if (j.contains("detected_objects") && j["detected_objects"].is_array()) {
			for (auto& obj : j["detected_objects"]) {
				Detection det;
				det.name       = obj.value("name", "unknown");
				det.confidence = obj.value("confidence", 0.0);
				det.x_min      = obj.value("x_min", 0);
				det.y_min      = obj.value("y_min", 0);
				det.x_max      = obj.value("x_max", 0);
				det.y_max      = obj.value("y_max", 0);
				result.objects.push_back(det);
			}
		}
	} catch (const json::exception& e) {
		std::cerr << "[JSON] Parse error: " << e.what() << std::endl;
	}

	return result;
}

// ── drawDetections ───────────────────────────────────────────────────
// Draw bounding boxes and labels on the ORIGINAL frame, scaling
// coordinates from the resized image back to the original resolution.
static void drawDetections(cv::Mat& frame,
						   const CloudResult& result)
{
	double scaleX = static_cast<double>(frame.cols) / result.imageWidth;
	double scaleY = static_cast<double>(frame.rows) / result.imageHeight;

	for (const auto& det : result.objects) {
		int x1 = static_cast<int>(det.x_min * scaleX);
		int y1 = static_cast<int>(det.y_min * scaleY);
		int x2 = static_cast<int>(det.x_max * scaleX);
		int y2 = static_cast<int>(det.y_max * scaleY);

		// Bounding box (green)
		cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2),
					  cv::Scalar(0, 255, 0), 2);

		// Label: object name only
		std::string label = det.name;

		int baseline = 0;
		cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
											0.5, 1, &baseline);
		int labelY = std::max(y1 - 6, textSize.height + 4);
		cv::rectangle(frame,
					  cv::Point(x1, labelY - textSize.height - 4),
					  cv::Point(x1 + textSize.width + 4, labelY + 2),
					  cv::Scalar(0, 255, 0), cv::FILLED);
		cv::putText(frame, label, cv::Point(x1 + 2, labelY - 2),
					cv::FONT_HERSHEY_SIMPLEX, 0.5,
					cv::Scalar(0, 0, 0), 1);
	}
}

// ── encodeToJpeg ─────────────────────────────────────────────────────
static bool encodeToJpeg(const cv::Mat& frame,
						 std::vector<uchar>& buffer)
{
	std::vector<int> params = { cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY };
	return cv::imencode(".jpg", frame, buffer, params);
}

#ifdef DEBUG_SAVE_FRAMES
static void debugSave(const cv::Mat& frame, const std::string& filename)
{
	std::filesystem::create_directories(DEBUG_OUTPUT_DIR);
	std::string path = std::string(DEBUG_OUTPUT_DIR) + "/" + filename;
	cv::imwrite(path, frame);
	std::cout << "[Debug] Saved " << path << std::endl;
}
#endif

// ── uploadFrame ──────────────────────────────────────────────────────
static std::string uploadFrame(const std::vector<uchar>& jpegBuffer,
							   const std::string& filename)
{
	std::string body(jpegBuffer.begin(), jpegBuffer.end());
	cpr::Response res = cpr::Post(
		cpr::Url{ API_ENDPOINT },
		cpr::Multipart{
			{ "file", cpr::Buffer{ body.begin(), body.end(), filename } }
		},
		cpr::Timeout{ 30000 }
	);

	if (res.status_code == 200) {
		std::cout << "[Upload] " << filename
				  << "  OK (" << jpegBuffer.size() << " bytes)" << std::endl;
		return res.text;
	}

	std::cerr << "[Upload] " << filename
			  << "  FAILED  status=" << res.status_code
			  << "  error=" << res.error.message << std::endl;
	return "";
}

// ── Main ─────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
	try {
		// --- Open video source ---
		cv::VideoCapture cap;
		bool isVideoFile = false;

		if (argc >= 2) {
			std::string videoPath = argv[1];
			cap.open(videoPath);
			isVideoFile = true;
			std::cout << "[DriveLens] Opening video file: " << videoPath << std::endl;
		} else {
			cap.open(0);
			std::cout << "[DriveLens] Opening webcam (device 0)" << std::endl;
		}

		if (!cap.isOpened()) {
			std::cerr << "[Error] Cannot open video source." << std::endl;
			return 1;
		}

		double fps = cap.get(cv::CAP_PROP_FPS);
		if (fps <= 0) fps = 30.0;

		int frameSkip = static_cast<int>(fps * CAPTURE_INTERVAL_SEC);
		std::cout << "[DriveLens] FPS: " << fps
				  << "  |  Capture every " << frameSkip << " frames ("
				  << CAPTURE_INTERVAL_SEC << "s)" << std::endl;

		// --- Main capture loop ---
		cv::Mat frame, displayFrame, resized;
		std::vector<uchar> jpegBuffer;
		int frameCount   = 0;
		int captureIndex = 0;

		// Last detection results – drawn on every frame until updated
		CloudResult lastDetection;

		// Async upload state – keeps video playing during HTTP POST
		std::future<std::string> pendingUpload;
		bool uploadInFlight = false;

		while (true) {
			if (!cap.read(frame) || frame.empty()) {
				if (isVideoFile)
					std::cout << "[DriveLens] End of video." << std::endl;
				else
					std::cerr << "[Error] Failed to read frame." << std::endl;
				break;
			}

			// Check if a background upload has finished (non-blocking)
			if (uploadInFlight &&
				pendingUpload.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
			{
				std::string cloudResponse = pendingUpload.get();
				uploadInFlight = false;

				lastDetection = parseCloudResponse(cloudResponse);

				if (!lastDetection.objects.empty()) {
					std::cout << "[Detect] " << lastDetection.objects.size()
							  << " object(s) found" << std::endl;
				}
			}

			// Draw detections on a COPY – keep original frame clean for upload
			displayFrame = frame.clone();
			if (!lastDetection.objects.empty()) {
				drawDetections(displayFrame, lastDetection);
			}

			cv::imshow("DriveLens Dashcam", displayFrame);
			if (cv::waitKey(1) == 27) break;

			++frameCount;
			if (frameCount % frameSkip != 0) continue;

			// Skip this capture if a previous upload is still in progress
			if (uploadInFlight) continue;

			// --- Resize the CLEAN frame for upload ---
			cv::resize(frame, resized, cv::Size(RESIZE_WIDTH, RESIZE_HEIGHT));

			// --- Encode to JPEG ---
			if (!encodeToJpeg(resized, jpegBuffer)) {
				std::cerr << "[Error] JPEG encode failed for frame "
						  << captureIndex << std::endl;
				continue;
			}

			std::string filename = "frame_" + std::to_string(captureIndex) + ".jpg";

#ifdef DEBUG_SAVE_FRAMES
			debugSave(resized, filename);
#endif

			// --- Launch upload in background thread ---
			std::vector<uchar> bufferCopy = jpegBuffer;
			pendingUpload = std::async(std::launch::async,
				uploadFrame, std::move(bufferCopy), filename);
			uploadInFlight = true;

			++captureIndex;
		}

		// Wait for any pending upload before cleanup
		if (uploadInFlight) {
			pendingUpload.wait();
		}

		cap.release();
		cv::destroyAllWindows();
		std::cout << "[DriveLens] Done. Uploaded " << captureIndex
				  << " frames." << std::endl;

	} catch (const std::exception& ex) {
		std::cerr << "[Fatal] " << ex.what() << std::endl;
		return 1;
	}

	return 0;
}
