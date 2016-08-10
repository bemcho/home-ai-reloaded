#pragma once
#include <string>
#include "ClipsAdapter.hpp"
using namespace std;
namespace hai
{
	class VisualREPL
	{
	public:
		VisualREPL(string replName, ClipsAdapter clips, bool showWindow);
		~VisualREPL();

	};
}
