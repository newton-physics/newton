#pragma once

//#ifndef AABB_H
//#define AABB_H

#include "btScalar.h"
#include "btVector3.h"
#include <vector>



// Implementation profiles
#define DBVT_IMPL_GENERIC		0	// Generic implementation	
#define DBVT_IMPL_SSE			1	// SSE

// Template implementation of ICollide
#ifdef _WIN32
#if (defined (_MSC_VER) && _MSC_VER >= 1400)
#define	DBVT_USE_TEMPLATE		1
#else
#define	DBVT_USE_TEMPLATE		0
#endif
#else
#define	DBVT_USE_TEMPLATE		0
#endif

// Use only intrinsics instead of inline asm
#define DBVT_USE_INTRINSIC_SSE	1

// Using memmov for collideOCL
#define DBVT_USE_MEMMOVE		1

// Enable benchmarking code
#define	DBVT_ENABLE_BENCHMARK	0

// Inlining
#define DBVT_INLINE				SIMD_FORCE_INLINE

// Specific methods implementation

//SSE gives errors on a MSVC 7.1
#if defined (BT_USE_SSE) //&& defined (_WIN32)
#define DBVT_SELECT_IMPL		DBVT_IMPL_SSE
#define DBVT_MERGE_IMPL			DBVT_IMPL_SSE
#define DBVT_INT0_IMPL			DBVT_IMPL_SSE
#else
#define DBVT_SELECT_IMPL		DBVT_IMPL_GENERIC
#define DBVT_MERGE_IMPL			DBVT_IMPL_GENERIC
#define DBVT_INT0_IMPL			DBVT_IMPL_GENERIC
#endif

#if	(DBVT_SELECT_IMPL==DBVT_IMPL_SSE)||	\
	(DBVT_MERGE_IMPL==DBVT_IMPL_SSE)||	\
	(DBVT_INT0_IMPL==DBVT_IMPL_SSE)
#include <emmintrin.h>
#endif

//
// Auto config and checks
//

#if DBVT_USE_TEMPLATE
#define	DBVT_VIRTUAL
#define DBVT_VIRTUAL_DTOR(a)
#define DBVT_PREFIX					template <typename T>
#define DBVT_IPOLICY				T& policy
#define DBVT_CHECKTYPE				static const ICollide&	typechecker=*(T*)1;(void)typechecker;
#else
#define	DBVT_VIRTUAL_DTOR(a)		virtual ~a() {}
#define DBVT_VIRTUAL				virtual
#define DBVT_PREFIX
#define DBVT_IPOLICY				ICollide& policy
#define DBVT_CHECKTYPE
#endif

#if DBVT_USE_MEMMOVE
#if !defined( __CELLOS_LV2__) && !defined(__MWERKS__)
#include <memory.h>
#endif
#include <string.h>
#endif

#ifndef DBVT_USE_TEMPLATE
#error "DBVT_USE_TEMPLATE undefined"
#endif

#ifndef DBVT_USE_MEMMOVE
#error "DBVT_USE_MEMMOVE undefined"
#endif

#ifndef DBVT_ENABLE_BENCHMARK
#error "DBVT_ENABLE_BENCHMARK undefined"
#endif

#ifndef DBVT_SELECT_IMPL
#error "DBVT_SELECT_IMPL undefined"
#endif

#ifndef DBVT_MERGE_IMPL
#error "DBVT_MERGE_IMPL undefined"
#endif

#ifndef DBVT_INT0_IMPL
#error "DBVT_INT0_IMPL undefined"
#endif


/* btDbvtAabbMm			*/
struct	btDbvtAabbMm
{
	DBVT_INLINE btVector3			Center() const	{ return((mi + mx) / 2); }
	DBVT_INLINE btVector3			Lengths() const	{ return(mx - mi); }
	DBVT_INLINE btVector3			Extents() const	{ return((mx - mi) / 2); }
	DBVT_INLINE const btVector3&	Mins() const	{ return(mi); }
	DBVT_INLINE const btVector3&	Maxs() const	{ return(mx); }
	static inline btDbvtAabbMm		FromCE(const btVector3& c, const btVector3& e);
	static inline btDbvtAabbMm		FromCR(const btVector3& c, btScalar r);
	static inline btDbvtAabbMm		FromMM(const btVector3& mi, const btVector3& mx);
	static inline btDbvtAabbMm		FromPoints(const btVector3* pts, int n);
	static inline btDbvtAabbMm		FromPoints(const btVector3** ppts, int n);
	DBVT_INLINE void				Expand(const btVector3& e);
	DBVT_INLINE void				SignedExpand(const btVector3& e);
	DBVT_INLINE bool				Contain(const btDbvtAabbMm& a) const;
	DBVT_INLINE int					Classify(const btVector3& n, btScalar o, int s) const;
	DBVT_INLINE btScalar			ProjectMinimum(const btVector3& v, unsigned signs) const;
	DBVT_INLINE friend bool			Intersect(const btDbvtAabbMm& a, const btDbvtAabbMm& b);


	DBVT_INLINE friend bool			Intersect(const btDbvtAabbMm& a, const btVector3& b);
	
	DBVT_INLINE friend bool			CheckSphere(const btDbvtAabbMm& a, const btVector3& e, const float radius);
	DBVT_INLINE friend bool			CheckRay(const btVector3& origin, const btVector3& dir);


	DBVT_INLINE friend btScalar		Proximity(const btDbvtAabbMm& a, const btDbvtAabbMm& b);
	DBVT_INLINE friend int			Select(const btDbvtAabbMm& o, const btDbvtAabbMm& a, const btDbvtAabbMm& b);
	DBVT_INLINE friend void			Merge(const btDbvtAabbMm& a, const btDbvtAabbMm& b, btDbvtAabbMm& r);
	DBVT_INLINE friend bool			NotEqual(const btDbvtAabbMm& a, const btDbvtAabbMm& b);

	DBVT_INLINE btVector3&	tMins()	{ return(mi); }
	DBVT_INLINE btVector3&	tMaxs()	{ return(mx); }

	//DBVT_INLINE bool			IntersectSphere(const btVector3& e, float radius);
//	DBVT_INLINE bool			CheckRay(const btVector3& origin, const btVector3& dir);
	DBVT_INLINE int			GetLongestExtent();
	DBVT_INLINE btScalar		Volume();

private:
	DBVT_INLINE void				AddSpan(const btVector3& d, btScalar& smi, btScalar& smx) const;
private:
	btVector3	mi, mx;
};

// Types	
typedef	btDbvtAabbMm	btDbvtVolume;

/* btDbvtNode				*/


//
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

//
DBVT_INLINE void		btDbvtAabbMm::Expand(const btVector3& e)
{
	mi -= e; mx += e;
}

//
DBVT_INLINE void		btDbvtAabbMm::SignedExpand(const btVector3& e)
{
	if (e.x()>0) mx.setX(mx.x() + e[0]); else mi.setX(mi.x() + e[0]);
	if (e.y()>0) mx.setY(mx.y() + e[1]); else mi.setY(mi.y() + e[1]);
	if (e.z()>0) mx.setZ(mx.z() + e[2]); else mi.setZ(mi.z() + e[2]);
}

//
DBVT_INLINE bool		btDbvtAabbMm::Contain(const btDbvtAabbMm& a) const
{
	return((mi.x() <= a.mi.x()) &&
		(mi.y() <= a.mi.y()) &&
		(mi.z() <= a.mi.z()) &&
		(mx.x() >= a.mx.x()) &&
		(mx.y() >= a.mx.y()) &&
		(mx.z() >= a.mx.z()));
}

//
DBVT_INLINE int		btDbvtAabbMm::Classify(const btVector3& n, btScalar o, int s) const
{
	btVector3			pi, px;
	switch (s)
	{
	case	(0 + 0 + 0) : px = btVector3(mi.x(), mi.y(), mi.z());
		pi = btVector3(mx.x(), mx.y(), mx.z()); break;
	case	(1 + 0 + 0) : px = btVector3(mx.x(), mi.y(), mi.z());
		pi = btVector3(mi.x(), mx.y(), mx.z()); break;
	case	(0 + 2 + 0) : px = btVector3(mi.x(), mx.y(), mi.z());
		pi = btVector3(mx.x(), mi.y(), mx.z()); break;
	case	(1 + 2 + 0) : px = btVector3(mx.x(), mx.y(), mi.z());
		pi = btVector3(mi.x(), mi.y(), mx.z()); break;
	case	(0 + 0 + 4) : px = btVector3(mi.x(), mi.y(), mx.z());
		pi = btVector3(mx.x(), mx.y(), mi.z()); break;
	case	(1 + 0 + 4) : px = btVector3(mx.x(), mi.y(), mx.z());
		pi = btVector3(mi.x(), mx.y(), mi.z()); break;
	case	(0 + 2 + 4) : px = btVector3(mi.x(), mx.y(), mx.z());
		pi = btVector3(mx.x(), mi.y(), mi.z()); break;
	case	(1 + 2 + 4) : px = btVector3(mx.x(), mx.y(), mx.z());
		pi = btVector3(mi.x(), mi.y(), mi.z()); break;
	}
	if ((btDot(n, px) + o)<0)		return(-1);
	if ((btDot(n, pi) + o) >= 0)	return(+1);
	return(0);
}

//
DBVT_INLINE btScalar	btDbvtAabbMm::ProjectMinimum(const btVector3& v, unsigned signs) const
{
	const btVector3*	b[] = { &mx, &mi };
	const btVector3		p(b[(signs >> 0) & 1]->x(),
		b[(signs >> 1) & 1]->y(),
		b[(signs >> 2) & 1]->z());
	return(btDot(p, v));
}

//
DBVT_INLINE void		btDbvtAabbMm::AddSpan(const btVector3& d, btScalar& smi, btScalar& smx) const
{
	for (int i = 0; i<3; ++i)
	{
		if (d[i]<0)
		{
			smi += mx[i] * d[i]; smx += mi[i] * d[i];
		}
		else
		{
			smi += mi[i] * d[i]; smx += mx[i] * d[i];
		}
	}
}

//
DBVT_INLINE bool		Intersect(const btDbvtAabbMm& a, const btDbvtAabbMm& b)
{
#if	DBVT_INT0_IMPL == DBVT_IMPL_SSE
	const __m128	rt(_mm_or_ps(_mm_cmplt_ps(_mm_load_ps(b.mx), _mm_load_ps(a.mi)),
		_mm_cmplt_ps(_mm_load_ps(a.mx), _mm_load_ps(b.mi))));
#if defined (_WIN32)
	const __int32*	pu((const __int32*)&rt);
#else
	const int*	pu((const int*)&rt);
#endif
	return((pu[0] | pu[1] | pu[2]) == 0);
#else
	return((a.mi.x() <= b.mx.x()) &&
		(a.mx.x() >= b.mi.x()) &&
		(a.mi.y() <= b.mx.y()) &&
		(a.mx.y() >= b.mi.y()) &&
		(a.mi.z() <= b.mx.z()) &&
		(a.mx.z() >= b.mi.z()));
#endif
}



//
DBVT_INLINE bool		Intersect(const btDbvtAabbMm& a, const btVector3& b)
{
	return((b.x() >= a.mi.x()) &&
		(b.y() >= a.mi.y()) &&
		(b.z() >= a.mi.z()) &&
		(b.x() <= a.mx.x()) &&
		(b.y() <= a.mx.y()) &&
		(b.z() <= a.mx.z()));
}

DBVT_INLINE bool		CheckSphere(const btDbvtAabbMm& a, const btVector3& bsCenter, const float radius)
{


	btVector3 center = a.Center();
	btVector3 extents = a.Extents();

	if (abs(center.x() - bsCenter.x()) <radius + extents.x() && abs(center.y() - bsCenter.y()) < radius + extents.y() && abs(center.z() - bsCenter.z()) < radius + extents.z())
	{
		return true;
	}

	return false;
}


SIMD_FORCE_INLINE bool		CheckRay(const btDbvtAabbMm& a, const btVector3& origin, const btVector3& dir)
{
	btScalar t = 0;
	// r.dir is unit direction vector of ray
	btVector3 dirfrac(1.0f / dir.x(), 1.0f / dir.y(), 1.0f / dir.z());
	btVector3 mi = a.Mins();
	btVector3 mx = a.Maxs();
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



//
DBVT_INLINE btScalar	Proximity(const btDbvtAabbMm& a, const btDbvtAabbMm& b)
{
	const btVector3	d = (a.mi + a.mx) - (b.mi + b.mx);
	return(btFabs(d.x()) + btFabs(d.y()) + btFabs(d.z()));
}



//
DBVT_INLINE int			Select(const btDbvtAabbMm& o, const btDbvtAabbMm& a, const btDbvtAabbMm& b)
{
#if	DBVT_SELECT_IMPL == DBVT_IMPL_SSE

#if defined (_WIN32)
	static ATTRIBUTE_ALIGNED16(const unsigned __int32)	mask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
#else
	static ATTRIBUTE_ALIGNED16(const unsigned int)	mask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x00000000 /*0x7fffffff*/ };
#endif
	///@todo: the intrinsic version is 11% slower
#if DBVT_USE_INTRINSIC_SSE

	union btSSEUnion ///NOTE: if we use more intrinsics, move btSSEUnion into the LinearMath directory
	{
		__m128		ssereg;
		float		floats[4];
		int			ints[4];
	};

	__m128	omi(_mm_load_ps(o.mi));
	omi = _mm_add_ps(omi, _mm_load_ps(o.mx));
	__m128	ami(_mm_load_ps(a.mi));
	ami = _mm_add_ps(ami, _mm_load_ps(a.mx));
	ami = _mm_sub_ps(ami, omi);
	ami = _mm_and_ps(ami, _mm_load_ps((const float*)mask));
	__m128	bmi(_mm_load_ps(b.mi));
	bmi = _mm_add_ps(bmi, _mm_load_ps(b.mx));
	bmi = _mm_sub_ps(bmi, omi);
	bmi = _mm_and_ps(bmi, _mm_load_ps((const float*)mask));
	__m128	t0(_mm_movehl_ps(ami, ami));
	ami = _mm_add_ps(ami, t0);
	ami = _mm_add_ss(ami, _mm_shuffle_ps(ami, ami, 1));
	__m128 t1(_mm_movehl_ps(bmi, bmi));
	bmi = _mm_add_ps(bmi, t1);
	bmi = _mm_add_ss(bmi, _mm_shuffle_ps(bmi, bmi, 1));

	btSSEUnion tmp;
	tmp.ssereg = _mm_cmple_ss(bmi, ami);
	return tmp.ints[0] & 1;

#else
	ATTRIBUTE_ALIGNED16(__int32	r[1]);
	__asm
	{
		mov		eax, o
			mov		ecx, a
			mov		edx, b
			movaps	xmm0, [eax]
			movaps	xmm5, mask
			addps	xmm0, [eax + 16]
			movaps	xmm1, [ecx]
			movaps	xmm2, [edx]
			addps	xmm1, [ecx + 16]
			addps	xmm2, [edx + 16]
			subps	xmm1, xmm0
			subps	xmm2, xmm0
			andps	xmm1, xmm5
			andps	xmm2, xmm5
			movhlps	xmm3, xmm1
			movhlps	xmm4, xmm2
			addps	xmm1, xmm3
			addps	xmm2, xmm4
			pshufd	xmm3, xmm1, 1
			pshufd	xmm4, xmm2, 1
			addss	xmm1, xmm3
			addss	xmm2, xmm4
			cmpless	xmm2, xmm1
			movss	r, xmm2
	}
	return(r[0] & 1);
#endif
#else
	return(Proximity(o, a)<Proximity(o, b) ? 0 : 1);
#endif
}

//
DBVT_INLINE void		Merge(const btDbvtAabbMm& a, const btDbvtAabbMm& b, btDbvtAabbMm& r)
{
#if DBVT_MERGE_IMPL==DBVT_IMPL_SSE
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

//
DBVT_INLINE bool		NotEqual(const btDbvtAabbMm& a, const btDbvtAabbMm& b)
{
	return((a.mi.x() != b.mi.x()) ||
		(a.mi.y() != b.mi.y()) ||
		(a.mi.z() != b.mi.z()) ||
		(a.mx.x() != b.mx.x()) ||
		(a.mx.y() != b.mx.y()) ||
		(a.mx.z() != b.mx.z()));
}

DBVT_INLINE btScalar		btDbvtAabbMm::Volume()
{
	btVector3 extents = (mx - mi) / 2;
	return (8.0f * extents.x() * extents.y() * extents.z());
}


DBVT_INLINE int		btDbvtAabbMm::GetLongestExtent()
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


//
//SIMD_FORCE_INLINE bool		btDbvtAabbMm::IntersectSphere(const btVector3& bsCenter, float radius)
//{
//	//return(	(b.x()>=a.mi.x())&&
//	//	(b.y()>=a.mi.y())&&
//	//	(b.z()>=a.mi.z())&&
//	//	(b.x()<=a.mx.x())&&
//	//	(b.y()<=a.mx.y())&&
//	//	(b.z()<=a.mx.z()));
//	btVector3 center = (mi + mx) / 2;
//	btVector3 extents = (mx - mi) / 2;
//
//	if (abs(center.x() - bsCenter.x()) <radius + extents.x() && abs(center.y() - bsCenter.y()) < radius + extents.y() && abs(center.z() - bsCenter.z()) < radius + extents.z())
//	{
//		return true;
//	}
//
//	return false;
//}

//SIMD_FORCE_INLINE bool		btDbvtAabbMm::IntersectRay(const btVector3& origin, const btVector3& dir)
//{
//	btScalar t = 0;
//	// r.dir is unit direction vector of ray
//	btVector3 dirfrac(1.0f / dir.x(), 1.0f / dir.y(), 1.0f / dir.z());
//
//	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
//	// r.org is origin of ray
//	btScalar t1 = (mi.x() - origin.x())*dirfrac.x();
//	btScalar t2 = (mx.x() - origin.x())*dirfrac.x();
//	btScalar t3 = (mi.y() - origin.y())*dirfrac.y();
//	btScalar t4 = (mx.y() - origin.y())*dirfrac.y();
//	btScalar t5 = (mi.z() - origin.z())*dirfrac.z();
//	btScalar t6 = (mx.z() - origin.z())*dirfrac.z();
//
//	btScalar tmin = btMax(btMax(btMin(t1, t2), btMin(t3, t4)), btMin(t5, t6));
//	btScalar tmax = btMin(btMin(btMax(t1, t2), btMax(t3, t4)), btMax(t5, t6));
//
//	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
//	if (tmax < 0)
//	{
//		t = tmax;
//		return false;
//	}
//
//	// if tmin > tmax, ray doesn't intersect AABB
//	if (tmin > tmax)
//	{
//		t = tmax;
//		return false;
//	}
//
//	t = tmin;
//	return true;
//}