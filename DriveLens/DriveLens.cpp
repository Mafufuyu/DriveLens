// DriveLens.cpp : Defines the entry point for the application.
//

#include "DriveLens.h"

using namespace std;

int main()
{
	// Verify OpenCV
	cout << "[DriveLens] OpenCV version: " << CV_VERSION << endl;

	// Verify cpr (HTTP client)
	cpr::Response r = cpr::Get(cpr::Url{"https://httpbin.org/get"});
	cout << "[DriveLens] HTTP status : " << r.status_code << endl;

	return 0;
}
