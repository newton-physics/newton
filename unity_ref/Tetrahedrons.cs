using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections.NotBurstCompatible;

namespace DefKit
{
    public unsafe class Tetrahedrons : MonoBehaviour
    {
        public int tetrasCount;

        public Tetrahedron[] tetras;

        //The first neighbor of the tetrahedron i is opposite to the first corner of tetrahedron i, and so on
        public TetrahedronNeighbours[] tetrasNeighbours;
        public float[] restVolumes;
        public int[] attributes;
        public int[] closestTriangles;

        //runtime arrays
        public NativeList<Tetrahedron> tetrasNative;
        public NativeList<TetrahedronNeighbours> tetrasNeighboursNative;
        public NativeList<byte> activeTetrasNative;
        public NativeList<float> restVolumesNative;
        public NativeList<int> closestTrianglesNative;

        public NativeList<Vector4> cnstrsDeltas;
        public NativeList<int> cnstrsCountNative;
        public NativeList<float> cnstrsMultiplierNative;

        public Tetrahedron* tetrasNativePtr;
        public void* tetrasNeighboursNativePtr;
        public float* restVolumesNativePtr;

        public void Awake()
        {
            InitRuntime();
        }

        public void InitRuntime()
        {
            InitNativeArrays();

            ResizeNativeArrays(this.tetrasCount);
            tetrasNative.CopyFromNBC(tetras);
            tetrasNeighboursNative.CopyFromNBC(tetrasNeighbours);
            restVolumesNative.CopyFromNBC(restVolumes);
            for (int i = 0; i < this.tetrasCount; i++)
            {
                activeTetrasNative[i] = 1;
            }

            if(closestTriangles != null && closestTriangles.Length > 0)
            {
                closestTrianglesNative = new NativeList<int>(tetrasCount, Allocator.Persistent);
                closestTrianglesNative.ResizeUninitialized(tetrasCount);
                closestTrianglesNative.CopyFromNBC(closestTriangles);
            }
        }

        public void InitManagedArrays(int count)
        {
            this.tetrasCount = count;
            this.tetras = new Tetrahedron[count];
            this.tetrasNeighbours = new TetrahedronNeighbours[count];
            this.restVolumes = new float[count];
            this.attributes = new int[count];
        }


        public void InitNativeArrays()
        {
            tetrasNative = new NativeList<Tetrahedron>(this.tetrasCount, Allocator.Persistent);
            tetrasNeighboursNative = new NativeList<TetrahedronNeighbours>(this.tetrasCount, Allocator.Persistent);
            restVolumesNative = new NativeList<float>(this.tetrasCount, Allocator.Persistent);
            activeTetrasNative = new NativeList<byte>(this.tetrasCount, Allocator.Persistent);

            cnstrsDeltas = new NativeList<Vector4>(this.tetrasCount * 4, Allocator.Persistent);
            cnstrsCountNative = new NativeList<int>(GetComponent<Body>().count, Allocator.Persistent);
            cnstrsMultiplierNative = new NativeList<float>(GetComponent<Body>().count, Allocator.Persistent);

            tetrasNativePtr = (Tetrahedron*)tetrasNative.GetUnsafePtr<Tetrahedron>();
            tetrasNeighboursNativePtr = tetrasNeighboursNative.GetUnsafePtr<TetrahedronNeighbours>();
            restVolumesNativePtr = (float*)restVolumesNative.GetUnsafePtr<float>();
        }


        public void ResizeNativeArrays(int newLength)
        {
            tetrasNative.ResizeUninitialized(newLength);
            tetrasNeighboursNative.ResizeUninitialized(newLength);
            activeTetrasNative.ResizeUninitialized(newLength);
            restVolumesNative.ResizeUninitialized(newLength);

            tetrasNativePtr = (Tetrahedron*)tetrasNative.GetUnsafePtr<Tetrahedron>();
            tetrasNeighboursNativePtr = tetrasNeighboursNative.GetUnsafePtr<TetrahedronNeighbours>();
            restVolumesNativePtr = (float*)restVolumesNative.GetUnsafePtr<float>();

            cnstrsDeltas.ResizeUninitialized(newLength * 4);
            cnstrsCountNative.ResizeUninitialized(GetComponent<Body>().count);
            cnstrsMultiplierNative.ResizeUninitialized(GetComponent<Body>().count);
        }

        public virtual void DisposeNativeArrays()
        {
            tetrasNative.Dispose();
            tetrasNeighboursNative.Dispose();
            restVolumesNative.Dispose();
            activeTetrasNative.Dispose();
            cnstrsDeltas.Dispose();
            cnstrsCountNative.Dispose();
            cnstrsMultiplierNative.Dispose();

            if (closestTrianglesNative.IsCreated)
                closestTrianglesNative.Dispose();

            tetrasNativePtr = null;
            tetrasNeighboursNativePtr = null;
            restVolumesNativePtr = null;
        }

        private void OnDestroy()
        {
            DisposeNativeArrays();
        }
    }
}