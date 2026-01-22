#ifndef __AABB_H__
#define __AABB_H__

#include "Common/Common.h"

namespace PBD
{
	class AABB
	{
	public:
		Vector3r m_p[2];

		AABB& operator = (const AABB& aabb)
		{ 
			m_p[0] = aabb.m_p[0]; 
			m_p[1] = aabb.m_p[1]; 
			return *this; 
		}

		static bool pointInAABB(const AABB& a, const Vector3r& p)
		{
			if ((p[0] < a.m_p[0][0]) || (p[1] < a.m_p[0][1]) || (p[2] < a.m_p[0][2]))
				return false;
			if ((p[0] > a.m_p[1][0]) || (p[1] > a.m_p[1][1]) || (p[2] > a.m_p[1][2]))
				return false;
			return true;
		}

		static void getEdge(const AABB& a, char i, Vector3r& p1, Vector3r& p2)
		{
			char c1, c2;
			getEdgeIndex(i, c1, c2);
			cornerPoint(a, c1, p1);
			cornerPoint(a, c2, p2);
		}

		static void getEdgeIndex(char i, char& p1, char& p2)
		{
			//                         0    1    2    3    4    5    6    7    8    9    10   11
			static char index[12*2] = {0,1, 0,2, 1,3, 2,3, 0,4, 1,5, 2,6, 3,7, 4,5, 4,6, 5,7, 6,7};
			p1 = index[2*i+0];
			p2 = index[2*i+1];
		}

		static void cornerPoint(const AABB& a, char i, Vector3r& p)
		{
			switch (i)
			{
			case 0:
				p = Vector3r(a.m_p[0][0], a.m_p[0][1], a.m_p[0][2]);
				break;
			case 1:
				p = Vector3r(a.m_p[1][0], a.m_p[0][1], a.m_p[0][2]);
				break;
			case 2:
				p = Vector3r(a.m_p[0][0], a.m_p[1][1], a.m_p[0][2]);
				break;
			case 3:
				p = Vector3r(a.m_p[1][0], a.m_p[1][1], a.m_p[0][2]);
				break;
			case 4:
				p = Vector3r(a.m_p[0][0], a.m_p[0][1], a.m_p[1][2]);
				break;
			case 5:
				p = Vector3r(a.m_p[1][0], a.m_p[0][1], a.m_p[1][2]);
				break;
			case 6:
				p = Vector3r(a.m_p[0][0], a.m_p[1][1], a.m_p[1][2]);
				break;
			case 7:
				p = Vector3r(a.m_p[1][0], a.m_p[1][1], a.m_p[1][2]);
				break;
			}
		}

		static FORCE_INLINE bool intersection(const AABB& a1, const AABB& a2)
		{
			for(char i=0;i<3;i++)
			{
				const Real min0 = a1.m_p[0][i];
				const Real max0 = a1.m_p[1][i];
				const Real min1 = a2.m_p[0][i];
				const Real max1 = a2.m_p[1][i];
				if (((max0 < min1) || (min0 > max1)))
					return false;
			}
			return true;
		}
	};
}

#endif
