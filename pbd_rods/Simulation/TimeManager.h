#ifndef _TIMEMANAGER_H
#define _TIMEMANAGER_H

#include "Common/Common.h"

namespace PBD
{
	class TimeManager
	{
	private:
		Real time;
		static TimeManager *current;
		Real h;

	public:
		TimeManager ();
		~TimeManager ();

		// Singleton
		static TimeManager* getCurrent ();
		static void setCurrent (TimeManager* tm);
		static bool hasCurrent();

		Real getTime();
		void setTime(Real t);
		Real getTimeStepSize();
		void setTimeStepSize(Real tss);
	};
}

#endif
