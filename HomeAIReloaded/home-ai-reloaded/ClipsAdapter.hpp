#pragma once
#include <string>
#include "opencv2/core.hpp"
#include <functional>
#include <memory>
#include "Annotation.hpp"
#include "clips/clips.h"
using namespace std;
namespace hai
{
	class ClipsAdapter
	{
	public:
		ClipsAdapter(const string aRulesFilePath) : rulesFilePath{ aRulesFilePath }
		{
			theCLIPSEnv = CreateEnvironment();
			EnvBuild(theCLIPSEnv, defaultDeftemplateFN().c_str());
			EnvLoad(theCLIPSEnv, aRulesFilePath.c_str());
		};
		~ClipsAdapter() { DestroyEnvironment(theCLIPSEnv); };

		inline void callFactCreateFN(Annotation annotation) noexcept { defaultFactCreateFN(annotation); };
		inline void callFactCreateFN(const vector<Annotation>& annotations) noexcept
		{
			for (const auto& a : annotations)
			{
				callFactCreateFN(a);
			}
		};

		inline	void envReset() noexcept { EnvReset(theCLIPSEnv); };
		inline	void envRun() noexcept { EnvRun(theCLIPSEnv, -1); };
		inline	void envEval(string clipsCommand, DATA_OBJECT& result) noexcept { EnvEval(theCLIPSEnv, clipsCommand.c_str(), &result); };

	private:
		cv::Ptr<void> theCLIPSEnv;
		string rulesFilePath;

		string defaultDeftemplateFN(void) noexcept
		{
			return "(deftemplate visualdetect"
					" (slot type (default object))"
					" (multislot at)"
					" (slot ontology)"
				" )";
		}

		inline void defaultFactCreateFN(Annotation annotation) noexcept { addDetectFact2(theCLIPSEnv, annotation.getType(), annotation.getRectangle(), annotation.getDescription()); }

		void addDetectFact2(void *environment, const string& type, const cv::Rect& at, const string& ontology) noexcept
		{
			void *newFact;
			void *templatePtr;
			void *theMultifield;
			DATA_OBJECT theValue;
			/*============================================================*/
			/* Disable garbage collection. It's only necessary to disable */
			/* garbage collection when calls are made into CLIPS from an */
			/* embedding program. It's not necessary to do this when the */
			/* the calls to user code are made by CLIPS (such as for */
			/* user-defined functions) or in the case of this example, */
			/* there are no calls to functions which can trigger garbage */
			/* collection (such as Send or FunctionCall). */
			/*============================================================*/
			//IncrementGCLocks(environment);
			/*==================*/
			/* Create the fact. */
			/*==================*/
			templatePtr = EnvFindDeftemplate(environment, "visualdetect");
			newFact = EnvCreateFact(environment, templatePtr);
			if (newFact == NULL) return;
			/*==============================*/
			/* Set the value of the type slot. */
			/*==============================*/
			theValue.type = SYMBOL;
			theValue.value = EnvAddSymbol(environment, type.c_str());
			EnvPutFactSlot(environment, newFact, "type", &theValue);
			/*==============================*/
			/* Set the value of the z slot. */
			/*==============================*/
			theMultifield = EnvCreateMultifield(environment, 4);
			SetMFType(theMultifield, 1, INTEGER);
			SetMFValue(theMultifield, 1, EnvAddLong(environment, at.x));

			SetMFType(theMultifield, 2, INTEGER);
			SetMFValue(theMultifield, 2, EnvAddLong(environment, at.y));

			SetMFType(theMultifield, 3, INTEGER);
			SetMFValue(theMultifield, 3, EnvAddLong(environment, at.width));

			SetMFType(theMultifield, 4, INTEGER);
			SetMFValue(theMultifield, 4, EnvAddLong(environment, at.height));


			SetDOBegin(theValue, 1);
			SetDOEnd(theValue, 4);
			theValue.type = MULTIFIELD;
			theValue.value = theMultifield;
			EnvPutFactSlot(environment, newFact, "at", &theValue);
			/*==============================*/
			/* Set the value of the what slot. */
			/*==============================*/
			theValue.type = SYMBOL;
			stringstream onto;
			onto << "\"" << ontology << "\"";
			theValue.value = EnvAddSymbol(environment, onto.str().c_str());
			EnvPutFactSlot(environment, newFact, "ontology", &theValue);
			/*=================================*/
			/* Assign default values since all */
			/* slots were not initialized. */
			/*=================================*/
			EnvAssignFactSlotDefaults(environment, newFact);
			/*==========================================================*/
			/* Enable garbage collection. Each call to IncrementGCLocks */
			/* should have a corresponding call to DecrementGCLocks. */
			/*==========================================================*/
			//EnvDecrementGCLocks(environment);
			/*==================*/
			/* Assert the fact. */
			/*==================*/
			EnvAssert(environment, newFact);
		}
	};
}
