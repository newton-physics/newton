using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace DefKit
{

    public unsafe class Body : MonoBehaviour
    {
        public static int globalCount = 0;

        public int globalInstanceId = -1;

        public int globalArrayId = -1;

        public int count = -1;

     //   [HideInInspector]
        public Vector4[] restPositions;

     //   [HideInInspector]
        public Vector4[] positions;

     //   [HideInInspector]
        public float[] masses;

      //  [HideInInspector]
        public float[] massesInv;

        public bool[] selectedPoints;

        public NativeList<Vector4> positionsNative;
        public NativeList<Vector4> restPositionsNative;
        public NativeList<Vector4> initialPositionsNative;

        public NativeList<Vector4> prevPositionsNative;

        public NativeList<Vector4> predictedPositionsNative;

        public NativeList<Vector4> velocitiesNative;
        public NativeList<Vector4> forcesNative;
        public NativeList<Vector4> tempNative;
        public NativeList<int> tempCounterNative;
        public NativeList<float>   massesInvNative;

        public NativeList<int> cnstrsCountNative;
        public NativeList<float> cnstrsMultiplierNative;


        public Vector4* restPositionsNativePtr;
        public Vector4* initialPositionsNativePtr;
        public Vector4* positionsNativePtr;
        public Vector4* predictedPositionsNativePtr;
        public float* massesInvNativePtr;
        public Vector4* velocitiesNativePtr;
        public Vector4* forcesNativePtr;

        public void Awake()
        {
           //Debug.Log("OnAwake:" + this.gameObject.name);
            InitRuntime();

            globalInstanceId = globalCount++;
        }

        public void OnEnable()
        {
            //ToDo runtime activation/deactivation handling
            //Debug.Log("OnEnable:" + this.gameObject.name);
        }


        public void OnDisable()
        {
            //Debug.Log("OnDisable:" + this.gameObject.name);
        }


        public virtual void InitRuntime()
        {
            InitNativeArrays();

            ResizeNativeArrays(this.count);

            for (int i = 0; i < this.count; i++)
            {
                this.restPositionsNative[i] = this.restPositions[i];

                 Vector4 pos = transform.TransformPoint(this.positions[i]);
                this.initialPositionsNative[i] = pos;
                this.positionsNative[i] = pos;
                this.predictedPositionsNative[i] = pos;
                this.prevPositionsNative[i] = pos;
                this.massesInvNative[i] = this.massesInv[i];
                this.velocitiesNative[i] = new Vector4();
                this.forcesNative[i] = new Vector4();
            }
        }

        public virtual void InitManagedArrays(int count)
        {
            this.count = count;
            this.masses = new float[count];
            this.massesInv = new float[count];
            this.restPositions = new Vector4[count];
            this.positions = new Vector4[count];
            this.selectedPoints = new bool[count];
        }

        public virtual void InitNativeArrays()
        {
     
            this.positionsNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.restPositionsNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.initialPositionsNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.predictedPositionsNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.prevPositionsNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.velocitiesNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.forcesNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.tempNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.tempCounterNative = new NativeList<int>(this.count, Allocator.Persistent);
            this.massesInvNative = new NativeList<float>(this.count, Allocator.Persistent);

            this.cnstrsCountNative = new NativeList<int>(this.count, Allocator.Persistent);
            this.cnstrsMultiplierNative = new NativeList<float>(this.count, Allocator.Persistent);

            this.initialPositionsNativePtr = (Vector4*)this.initialPositionsNative.GetUnsafePtr<Vector4>();
            this.restPositionsNativePtr = (Vector4*)this.restPositionsNative.GetUnsafePtr<Vector4>();
            this.positionsNativePtr = (Vector4*)this.positionsNative.GetUnsafePtr<Vector4>();
            this.predictedPositionsNativePtr = (Vector4*)this.predictedPositionsNative.GetUnsafePtr<Vector4>();
            this.velocitiesNativePtr = (Vector4*)this.velocitiesNative.GetUnsafePtr<Vector4>();
            this.massesInvNativePtr = (float*)this.massesInvNative.GetUnsafePtr<float>();
            this.forcesNativePtr = (Vector4*)this.forcesNative.GetUnsafePtr<Vector4>();

        }

        public void ResizeNativeArrays(int newLength)
        {
            this.positionsNative.ResizeUninitialized(newLength);
            this.initialPositionsNative.ResizeUninitialized(newLength);
            this.restPositionsNative.ResizeUninitialized(newLength);
            this.predictedPositionsNative.ResizeUninitialized(newLength);
            this.prevPositionsNative.ResizeUninitialized(newLength);
            this.massesInvNative.ResizeUninitialized(newLength);
            this.velocitiesNative.ResizeUninitialized(newLength);
            this.forcesNative.ResizeUninitialized(newLength);
            this.cnstrsCountNative.ResizeUninitialized(newLength);
            this.cnstrsMultiplierNative.ResizeUninitialized(newLength);
            this.tempNative.ResizeUninitialized(newLength);
            this.tempCounterNative.ResizeUninitialized(newLength);

            this.restPositionsNativePtr = (Vector4*)this.restPositionsNative.GetUnsafePtr<Vector4>();
            this.initialPositionsNativePtr = (Vector4*)this.initialPositionsNative.GetUnsafePtr<Vector4>();
            this.positionsNativePtr = (Vector4*)this.positionsNative.GetUnsafePtr<Vector4>();
            this.predictedPositionsNativePtr = (Vector4*)this.predictedPositionsNative.GetUnsafePtr<Vector4>();
            this.velocitiesNativePtr = (Vector4*)this.velocitiesNative.GetUnsafePtr<Vector4>();
            this.forcesNativePtr = (Vector4*)this.forcesNative.GetUnsafePtr<Vector4>();
            this.massesInvNativePtr = (float*)this.massesInvNative.GetUnsafePtr<float>();
        }

        public void ResetSimulation()
        {

            // this.massesInvNative.CopyFrom(this.massesInv);
            for (int i = 0; i < this.count; i++)
            {
                Vector4 pos = transform.TransformPoint(this.restPositions[i]);
                this.positions[i] = pos;
                this.positionsNative[i] = pos;
                this.velocitiesNative[i] = default(Vector4);
                this.forcesNative[i] = default(Vector4);
            }
        }


        public virtual void DisposeArrays()
        {
            this.massesInvNative.Dispose();
            this.positionsNative.Dispose();
            this.restPositionsNative.Dispose();
            this.initialPositionsNative.Dispose();
            this.predictedPositionsNative.Dispose();
            this.prevPositionsNative.Dispose();
            this.velocitiesNative.Dispose();
            this.forcesNative.Dispose();
            this.tempNative.Dispose();

            this.cnstrsCountNative.Dispose();
            this.cnstrsMultiplierNative.Dispose();
            this.tempCounterNative.Dispose();

            this.restPositionsNativePtr = null;
            this.positionsNativePtr = null;
            this.predictedPositionsNativePtr = null;
            this.velocitiesNativePtr = null;
            this.massesInvNativePtr = null;
            this.forcesNativePtr = null;
        }

        void OnDestroy()
        {
            DisposeArrays();
        }
    }

}