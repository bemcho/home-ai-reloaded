#pragma once
#include <string>
#include "opencv2/core.hpp"
#include <functional>
#include <memory>
#include "Annotation.hpp"
#include "clips/clips.h"
#include "tbb/mutex.h"
#include <chrono>
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
			EnvReset(theCLIPSEnv);
		};
		~ClipsAdapter() { DestroyEnvironment(theCLIPSEnv); };

		inline void callFactCreateFN(Annotation& annotation, const string& visualRepl) noexcept { tbb::mutex::scoped_lock(createOneFactLock); addDetectFact2(theCLIPSEnv, annotation, visualRepl); };
		inline void callFactCreateFN(vector<Annotation>& annotations, const string& visualRepl) noexcept
		{
			tbb::mutex::scoped_lock(createOneFactLock);
			for (auto& a : annotations)
			{
				addDetectFact2(theCLIPSEnv, a, visualRepl);
			}
		};

		inline	void envReset() noexcept { tbb::mutex::scoped_lock(globalEnvLock); EnvReset(theCLIPSEnv); };
		inline	void envRun() noexcept { tbb::mutex::scoped_lock(globalEnvLock); EnvRun(theCLIPSEnv, -1); };
		inline	void envEval(string clipsCommand, DATA_OBJECT& result) noexcept { tbb::mutex::scoped_lock(globalEnvLock); EnvEval(theCLIPSEnv, clipsCommand.c_str(), &result); };
		inline	void envClear() noexcept { tbb::mutex::scoped_lock(globalEnvLock); EnvClear(theCLIPSEnv); };

	private:
		tbb::mutex createOneFactLock, createFactsLock,globalEnvLock;
		cv::Ptr<void> theCLIPSEnv;
		string rulesFilePath;

		string defaultDeftemplateFN(void) noexcept
		{
			return "(deftemplate visualdetect"
				" (slot type (default object))"
				" (multislot rectangle)"
				" (slot ontology)"
				" (slot at)"
				" (slot timestamp)"
				" )";
		}

		void addDetectFact2(void *environment, Annotation& a, const string& visualRepl) noexcept
		{
			tbb::mutex::scoped_lock(createOneFactLock);
			if (a.getType().compare("empty") == 0)
				return;
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
			/* Set the value of the 'type' slot. */
			/*==============================*/
			theValue.type = SYMBOL;
			theValue.value = EnvAddSymbol(environment, a.getType().c_str());
			EnvPutFactSlot(environment, newFact, "type", &theValue);
			/*==============================*/
			/* Set the value of the 'rectangle' slot. */
			/*==============================*/

			cv::Rect at;
			if (a.getType().compare("contour") == 0)
			{
				at = boundingRect(a.getContour());
			}
			else
			{
				at = a.getRectangle();
			}

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
			EnvPutFactSlot(environment, newFact, "rectangle", &theValue);
			/*==============================*/
			/* Set the value of the what 'onthology' slot. */
			/*==============================*/
			theValue.type = SYMBOL;
			stringstream onto;
			onto << "\"" << a.getDescription() << "\"";
			theValue.value = EnvAddSymbol(environment, onto.str().c_str());
			EnvPutFactSlot(environment, newFact, "ontology", &theValue);
			/*==============================*/
			/* Set the value of the what  'at' slot. */
			/*==============================*/
			theValue.type = SYMBOL;
			stringstream repl;
			repl << "\"" << visualRepl << "\"";
			theValue.value = EnvAddSymbol(environment, repl.str().c_str());
			EnvPutFactSlot(environment, newFact, "at", &theValue);
			/*==============================*/
			/* Set the value of the what 'timestamp' slot. */
			/*==============================*/
			theValue.type = INTEGER;
			theValue.value = EnvAddLong(environment, std::chrono::duration_cast<std::chrono::nanoseconds>(
				std::chrono::system_clock::now().time_since_epoch()).count());
			EnvPutFactSlot(environment, newFact, "timestamp", &theValue);
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
