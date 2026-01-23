#pragma once

//#ifndef AABB_H
//#define AABB_H

#include "btScalar.h"
#include "btVector3.h"
#include <vector>

//struct	btDbvtAabbMm
ATTRIBUTE_ALIGNED16(struct) btDbvtAabbMm
{
	SIMD_FORCE_INLINE btVector3			Center() const	{ return((mi + mx) / 2); }
	SIMD_FORCE_INLINE btVector3			Lengths() const	{ return(mx - mi); }
	SIMD_FORCE_INLINE btVector3			Extents() const	{ return((mx - mi) / 2); }
	SIMD_FORCE_INLINE const btVector3&	Mins() const	{ return(mi); }
	SIMD_FORCE_INLINE const btVector3&	Maxs() const	{ return(mx); }

	static inline btDbvtAabbMm		FromCE(const btVector3& c, const btVector3& e);
	static inline btDbvtAabbMm		FromCR(const btVector3& c, btScalar r);
	static inline btDbvtAabbMm		FromMM(const btVector3& mi, const btVector3& mx);
	static inline btDbvtAabbMm		FromPoints(const btVector3* pts, int n);
	static inline btDbvtAabbMm		FromPoints(const btVector3** ppts, int n);

	SIMD_FORCE_INLINE void				Expand(const btVector3& e);
	SIMD_FORCE_INLINE void				SignedExpand(const btVector3& e);
	SIMD_FORCE_INLINE bool				Contain(const btDbvtAabbMm& a) const;
	SIMD_FORCE_INLINE int					Classify(const btVector3& n, btScalar o, int s) const;
	SIMD_FORCE_INLINE btScalar			ProjectMinimum(const btVector3& v, unsigned signs) const;
	SIMD_FORCE_INLINE friend bool			Intersect(const btDbvtAabbMm& a, const btDbvtAabbMm& b);
	SIMD_FORCE_INLINE friend bool			Intersect(const btDbvtAabbMm& a, const btVector3& b);
	SIMD_FORCE_INLINE friend btScalar		Proximity(const btDbvtAabbMm& a, const btDbvtAabbMm& b);
	SIMD_FORCE_INLINE friend int			Select(const btDbvtAabbMm& o, const btDbvtAabbMm& a, const btDbvtAabbMm& b);
	SIMD_FORCE_INLINE friend void			Merge(const btDbvtAabbMm& a, const btDbvtAabbMm& b, btDbvtAabbMm& r);
	SIMD_FORCE_INLINE friend bool			NotEqual(const btDbvtAabbMm& a, const btDbvtAabbMm& b);

	SIMD_FORCE_INLINE btVector3&	tMins()	{ return(mi); }
	SIMD_FORCE_INLINE btVector3&	tMaxs()	{ return(mx); }

	SIMD_FORCE_INLINE void			Union(const btDbvtAabbMm& b);
	SIMD_FORCE_INLINE btScalar		Volume();
	SIMD_FORCE_INLINE bool			IntersectSphere(const btVector3& e, float radius);
	SIMD_FORCE_INLINE bool			IntersectRay(const btVector3& origin, const btVector3& dir);
	SIMD_FORCE_INLINE int			GetLongestExtent();

private:
	btVector3	mi, mx;

};

inline btDbvtAabbMm			btDbvtAabbMm::FromCE(const btVector3& c, const btVector3& e)
{
	btDbvtAabbMm box;
	box.mi = c - e; box.mx = c + e;
	return(box);
}

//
inline btDbvtAabbMm			btDbvtAabbMm::FromCR(const btVector3& c, btScalar r)
{
	return(FromCE(c, btVector3(r, r, r)));
}

//
inline btDbvtAabbMm			btDbvtAabbMm::FromMM(const btVector3& mi, const btVector3& mx)
{
	btDbvtAabbMm box;
	box.mi = mi; box.mx = mx;
	return(box);
}

//
inline btDbvtAabbMm			btDbvtAabbMm::FromPoints(const btVector3* pts, int n)
{
	btDbvtAabbMm box;
	box.mi = box.mx = pts[0];
	for (int i = 1; i<n; ++i)
	{
		box.mi.setMin(pts[i]);
		box.mx.setMax(pts[i]);
	}
	return(box);
}

//
inline btDbvtAabbMm			btDbvtAabbMm::FromPoints(const btVector3** ppts, int n)
{
	btDbvtAabbMm box;
	box.mi = box.mx = *ppts[0];
	for (int i = 1; i<n; ++i)
	{
		box.mi.setMin(*ppts[i]);
		box.mx.setMax(*ppts[i]);
	}
	return(box);
}

SIMD_FORCE_INLINE void		btDbvtAabbMm::Union(const btDbvtAabbMm& b)
{
#ifdef BT_USE_SSE
	__m128	ami(_mm_load_ps(mi));
	__m128	amx(_mm_load_ps(mx));
	__m128	bmi(_mm_load_ps(b.mi));
	__m128	bmx(_mm_load_ps(b.mx));
	ami = _mm_min_ps(ami, bmi);
	amx = _mm_max_ps(amx, bmx);
	_mm_store_ps(mi, ami);
	_mm_store_ps(mx, amx);
#else
	for (int i = 0; i<3; ++i)
	{
		if (mi[i]<b.mi[i])
			mi[i] = mi[i];
		else
			r.mi[i] = b.mi[i];

		if (mx[i]>b.mx[i])
			mx[i] = mx[i];
		else
			mx[i] = b.mx[i];
	}
#endif
}


//
SIMD_FORCE_INLINE btScalar		btDbvtAabbMm::Volume()
{
	btVector3 extents = (mx - mi) / 2;
	return (8.0f * extents.x() * extents.y() * extents.z());
}

//
SIMD_FORCE_INLINE void		btDbvtAabbMm::Expand(const btVector3& e)
{
	mi -= e;
	mx += e;
}

//
SIMD_FORCE_INLINE void		btDbvtAabbMm::SignedExpand(const btVector3& e)
{
	if (e.x()>0) mx.setX(mx.x() + e[0]); else mi.setX(mi.x() + e[0]);
	if (e.y()>0) mx.setY(mx.y() + e[1]); else mi.setY(mi.y() + e[1]);
	if (e.z()>0) mx.setZ(mx.z() + e[2]); else mi.setZ(mi.z() + e[2]);
}

//
SIMD_FORCE_INLINE bool		btDbvtAabbMm::Contain(const btDbvtAabbMm& a) const
{
	return((mi.x() <= a.mi.x()) && (mi.y() <= a.mi.y()) && (mi.z() <= a.mi.z()) && (mx.x() >= a.mx.x()) && (mx.y() >= a.mx.y()) && (mx.z() >= a.mx.z()));
}

SIMD_FORCE_INLINE bool		Intersect(const btDbvtAabbMm& a, const btVector3& b)
{
	return((b.x() >= a.mi.x()) && (b.y() >= a.mi.y()) && (b.z() >= a.mi.z()) && (b.x() <= a.mx.x()) && (b.y() <= a.mx.y()) && (b.z() <= a.mx.z()));
}

SIMD_FORCE_INLINE void		Merge(const btDbvtAabbMm& a, const btDbvtAabbMm& b, btDbvtAabbMm& r)
{
#ifdef BT_USE_SSE
	__m128	ami(_mm_load_ps(a.mi));
	__m128	amx(_mm_load_ps(a.mx));
	__m128	bmi(_mm_load_ps(b.mi));
	__m128	bmx(_mm_load_ps(b.mx));
	ami = _mm_min_ps(ami, bmi);
	amx = _mm_max_ps(amx, bmx);
	_mm_store_ps(r.mi, ami);
	_mm_store_ps(r.mx, amx);
#else
	for (int i = 0; i<3; ++i)
	{
		if (a.mi[i]<b.mi[i]) r.mi[i] = a.mi[i]; else r.mi[i] = b.mi[i];
		if (a.mx[i]>b.mx[i]) r.mx[i] = a.mx[i]; else r.mx[i] = b.mx[i];
	}
#endif
}


SIMD_FORCE_INLINE bool		btDbvtAabbMm::IntersectSphere(const btVector3& bsCenter, float radius)
{
	//return(	(b.x()>=a.mi.x())&&
	//	(b.y()>=a.mi.y())&&
	//	(b.z()>=a.mi.z())&&
	//	(b.x()<=a.mx.x())&&
	//	(b.y()<=a.mx.y())&&
	//	(b.z()<=a.mx.z()));
	btVector3 center = (mi + mx) / 2;
	btVector3 extents = (mx - mi) / 2;

	if (abs(center.x() - bsCenter.x()) <radius + extents.x() && abs(center.y() - bsCenter.y()) < radius + extents.y() && abs(center.z() - bsCenter.z()) < radius + extents.z())
	{
		return true;
	}

	return false;
}

SIMD_FORCE_INLINE bool		btDbvtAabbMm::IntersectRay(const btVector3& origin, const btVector3& dir)
{
	btScalar t = 0;
	// r.dir is unit direction vector of ray
	btVector3 dirfrac(1.0f / dir.x(), 1.0f / dir.y(), 1.0f / dir.z());

	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	// r.org is origin of ray
	btScalar t1 = (mi.x() - origin.x())*dirfrac.x();
	btScalar t2 = (mx.x() - origin.x())*dirfrac.x();
	btScalar t3 = (mi.y() - origin.y())*dirfrac.y();
	btScalar t4 = (mx.y() - origin.y())*dirfrac.y();
	btScalar t5 = (mi.z() - origin.z())*dirfrac.z();
	btScalar t6 = (mx.z() - origin.z())*dirfrac.z();

	btScalar tmin = btMax(btMax(btMin(t1, t2), btMin(t3, t4)), btMin(t5, t6));
	btScalar tmax = btMin(btMin(btMax(t1, t2), btMax(t3, t4)), btMax(t5, t6));

	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
	if (tmax < 0)
	{
		t = tmax;
		return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tmin > tmax)
	{
		t = tmax;
		return false;
	}

	t = tmin;
	return true;
}


SIMD_FORCE_INLINE int		btDbvtAabbMm::GetLongestExtent()
{
	btVector3 extent = (mx - mi) / 2;


	if (extent.x() >= extent.y() && extent.x() >= extent.z())
		return 0;
	else if (extent.y() >= extent.x() && extent.y() >= extent.z())
		return 1;
	else if (extent.z() >= extent.x() && extent.z() >= extent.y())
		return 2;

	return 0;
}

typedef	btDbvtAabbMm	AABB;

//#endif //AABB_TREE_H