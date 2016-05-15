#include "Annotation.h"
#include<string>
using namespace std;


Annotation::Annotation()
{
}

Annotation::Annotation(cv::Rect rect, std::string desc) : description(desc), rectangle(rect)
{
}


Annotation::~Annotation()
{
}

string Annotation::getDescription()
{
	return description;
}

cv::Rect Annotation::getRectangle()
{
	return rectangle;
}

void Annotation::setDescription(std::string desc)
{
	description = desc;
}

void Annotation::setRectangle(cv::Rect rect)
{
	rectangle = rect;
}


