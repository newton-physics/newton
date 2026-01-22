#ifndef __IDFACTORY_H__
#define __IDFACTORY_H__

namespace PBD
{
	/** Factory for unique ids.
	  */
	class IDFactory
	{
	private:
		/** Current id */
		static int id;

	public:
		static int getId() { return id++; }
	};
}

#endif

