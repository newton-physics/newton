
using UnityEngine;
using Unity.Collections;

using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections.NotBurstCompatible;

namespace DefKit
{
    public unsafe class Edges : MonoBehaviour
    {

        public int edgeCount;
        public Edge[] edges;
        public int[] sharedTriangles;
        public int[] batchGroupId;
        public int batchGroupCount;

        public NativeList<Edge> edgesNative;

        public Edge* edgesNativePtr;

        public NativeList<Vector4> cnstrsDeltas;
        public NativeList<int> cnstrsCountNative;
        public NativeList<float> cnstrsMultiplierNative;

        public NativeList<float> lambdasNative;
        public NativeList<float> lambdasBNative;
        public void Awake()
        {
            InitRuntime();
        }

        public void InitRuntime()
        {
            InitNativeArrays();
            ResizeNativeArrays(edgeCount);
            edgesNative.CopyFromNBC(edges);
        }


        public void InitManagedArrays(int edgesCount)
        {
            edgeCount = edgesCount;
            edges = new Edge[edgesCount];
            batchGroupId = new int[edgesCount];
            sharedTriangles = new int[edgesCount * 2];
        }

        public void InitNativeArrays()
        {
            edgesNative = new NativeList<Edge>(this.edgeCount, Allocator.Persistent);
            cnstrsDeltas = new NativeList<Vector4>(this.edgeCount * 2, Allocator.Persistent);
            cnstrsCountNative = new NativeList<int>(GetComponent<Body>().count, Allocator.Persistent);
            cnstrsMultiplierNative = new NativeList<float>(GetComponent<Body>().count, Allocator.Persistent);


            lambdasNative = new NativeList<float>(this.edgeCount, Allocator.Persistent);
            lambdasBNative = new NativeList<float>(this.edgeCount, Allocator.Persistent);

            edgesNativePtr = (Edge*)edgesNative.GetUnsafePtr<Edge>();
        }


        public bool AddSharedTri(int edgeId, int triId)
        {
            if (sharedTriangles[edgeId * 2 + 0] == -1)
            {
                sharedTriangles[edgeId * 2 + 0] = triId;
                return true;
            }
            else if (sharedTriangles[edgeId * 2 + 1] == -1)
            {
                // check tri not referencing same edge
                if (triId == sharedTriangles[edgeId * 2 + 0])
                    return false;
                else
                {
                    sharedTriangles[edgeId * 2 + 1] = triId;
                    return true;
                }
            }
            else
                return false;
        }

        public void ResizeNativeArrays(int newLength)
        {
            edgesNative.ResizeUninitialized(newLength);
            lambdasNative.ResizeUninitialized(newLength);
            lambdasBNative.ResizeUninitialized(newLength);
            cnstrsDeltas.ResizeUninitialized(newLength * 2);
            cnstrsCountNative.ResizeUninitialized(GetComponent<Body>().count);
            cnstrsMultiplierNative.ResizeUninitialized(GetComponent<Body>().count);
        }

        public virtual void DisposeNativeArrays()
        {

            this.edgesNative.Dispose();
            this.cnstrsDeltas.Dispose();
            this.cnstrsCountNative.Dispose();
            this.cnstrsMultiplierNative.Dispose();
            this.lambdasNative.Dispose();
            this.lambdasBNative.Dispose();
            edgesNativePtr = null;
        }

        private void OnDestroy()
        {
            DisposeNativeArrays();
        }
    }
}