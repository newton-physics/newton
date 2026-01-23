using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections.NotBurstCompatible;

namespace DefKit
{
    public unsafe class Triangles : MonoBehaviour
    {
        public int globalArrayId = -1;
        public int trianglesCount;
        public int prevTrianglesCount;
        public int trianglesNeighboursCount;

        public Triangle[] triangles;
        public int[] trianglesMarkers;
        public int[] trianglesNeighbours;


        //Runtime arrays
        public NativeList<Triangle> trianglesNative;
        public NativeList<Triangle> prevTrianglesNative;

        public NativeList<int> trianglesNeighboursNative;

        public Triangle* trianglesNativePtr;
        public int* trianglesNeighboursNativePtr;


        public void Awake()
        {
            InitRuntime();
        }

        public void InitRuntime()
        {
            InitNativeArrays();

            ResizeNativeArrays(this.trianglesCount);

            trianglesNative.CopyFromNBC(triangles);
            prevTrianglesNative.CopyFromNBC(triangles);
            //trianglesNeighboursNative.CopyFromNBC(trianglesNeighbours);
        }

        public void InitManagedArrays(int count, int neighCount)
        {
            trianglesCount = count;
            prevTrianglesCount = count;
            triangles = new Triangle[count];
            trianglesMarkers = new int[count];
            trianglesNeighboursCount = neighCount;
            trianglesNeighbours = new int[neighCount];
        }


        public void InitNativeArrays()
        {
            trianglesNative = new NativeList<Triangle>(this.trianglesCount, Allocator.Persistent);
            prevTrianglesNative = new NativeList<Triangle>(this.trianglesCount, Allocator.Persistent);
            trianglesNeighboursNative = new NativeList<int>(this.trianglesNeighboursCount, Allocator.Persistent);

            UpdateNativePointers();
        }

        public void ResizeNativeArrays(int newLength)
        {
            trianglesNative.ResizeUninitialized(newLength);
            prevTrianglesNative.ResizeUninitialized(newLength);
            trianglesNeighboursNative.ResizeUninitialized(newLength);

            UpdateNativePointers();
        }

        public void UpdateNativePointers()
        {
            trianglesNativePtr = (Triangle*)trianglesNative.GetUnsafePtr<Triangle>();
            trianglesNeighboursNativePtr = (int*)trianglesNeighboursNative.GetUnsafePtr<int>();
        }


        public virtual void DisposeNativeArrays()
        {

            trianglesNative.Dispose();
            prevTrianglesNative.Dispose();
            trianglesNeighboursNative.Dispose();
            trianglesNativePtr = null;
            trianglesNeighboursNativePtr = null;

        }

        private void OnDestroy()
        {
            DisposeNativeArrays();
        }
    }
}