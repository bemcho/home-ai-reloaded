#include "Annotation.h"
#include<string>
using namespace std;


Annotation::Annotation()
{
}

Annotation::Annotation(cv::Rect rect, std::string desc, std::string annotationType) : description(desc), rectangle(rect),type(annotationType)
{
}


Annotation::~Annotation()
{
}

string Annotation::getDescription()
{
	return description;
}

std::string Annotation::getType()
{
	return type;
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


