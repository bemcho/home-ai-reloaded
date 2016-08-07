#pragma once
#include<string>
#include <opencv2/core.hpp>
namespace hai
{
	class Annotation
	{
	public:
		Annotation() {}
		Annotation(cv::Rect rect, std::string desc, std::string type) : description(desc), rectangle(rect), type(type){}
		virtual ~Annotation() {};
		std::string getDescription() { return description; }
		std::string getType() { return type; }
		cv::Rect getRectangle() { return rectangle;}
		void setDescription(std::string desc) { description = desc; }
		void setRectangle(cv::Rect rect) { rectangle = rect; }
		void setType(std::string type) { this->type = type; }
	private:
		cv::Rect rectangle;
		std::string description;
		std::string type;
	};
}

