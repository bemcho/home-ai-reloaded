#pragma once
#include<string>
#include <opencv2/core.hpp>
namespace hai
{
	class Annotation
	{
	public:
		Annotation() noexcept : rectangle{ cv::Rect(0,0,0,0)}, description{ "empty" }, type{"empty"} {};
		Annotation(std::vector<cv::Point> aContour, std::string desc, std::string type) noexcept : description{ desc }, contour{ aContour }, type{ type } {};
		Annotation(cv::Rect rect, std::string desc, std::string type) : description{ desc }, rectangle{ rect }, type{ type } {};
		~Annotation() {};

		Annotation& operator=(Annotation& other) = default;

		inline const std::string getDescription() noexcept { return description; };
		inline const std::string getType() noexcept { return type; };
		inline const cv::Rect getRectangle() noexcept { return rectangle; };
		inline const std::vector<cv::Point> getContour() noexcept { return contour; };

	private:
		cv::Rect rectangle;
		std::string description;
		std::vector<cv::Point> contour;
		std::string type;
	};
}

