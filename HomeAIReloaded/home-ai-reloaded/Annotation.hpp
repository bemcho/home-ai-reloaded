#pragma once
#include<string>
#include <opencv2/core.hpp>
namespace hai
{
	class Annotation
	{
	public:
		Annotation() {};
		Annotation(cv::Rect rect, std::string desc, std::string type) : description{ desc }, rectangle{ rect }, type{ type } {};
		~Annotation() {};

		Annotation& operator=(Annotation& other) = default;

		inline const std::string getDescription() noexcept { return description; };
		inline const std::string getType() noexcept { return type; };
		inline const cv::Rect getRectangle() noexcept { return rectangle; };

	private:
		cv::Rect rectangle;
		std::string description;
		std::string type;
	};
}

