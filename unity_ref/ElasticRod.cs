/*---------------------------------------------------------------------
* Copyright(C) 2017-2020 KRN Labs 
* Dr Przemyslaw Korzeniowski <korzenio@gmail.com>
* All rights reserved.
*
* This file is part of DefKit.
* It can not be copied and/or distributed 
* without the express permission of Dr Przemyslaw Korzeniowski
* -------------------------------------------------------------------*/

using System;
using UnityEngine;
using Unity.Collections;

using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections.NotBurstCompatible;

namespace DefKit.ElasticRods
{
    public unsafe class ElasticRod : MonoBehaviour
    {
        public enum ElasticRodSolverType
        {
            ITERATIVE,
            DIRECT_REFERENCE,
            DIRECT_BANDED,
            DIRECT_TASSILO
        }

        public ElasticRodSolverType solverType;

        public int size = -1;

        public float radius = 1.0f;

        public float restLength = 1.0f;

        public float youngModulus = 1000000f;

        public float torsionModulus = 1000000f;

        public IntPtr rodPtr;
        public IntPtr rodAdvPtr;

        public Quaternion[] restOrientations;
        public Quaternion[] orientations;

        public Vector4[] intrinsicBend;
        public Vector4[] intrinsicBendKs;

        public float[] quatMassesInv;

        public float[] restLenghts;

        public NativeList<Quaternion> orientationsNative;
        public NativeList<Quaternion> prevOrientationsNative;
        public NativeList<Quaternion> predictedOrientationsNative;

        public NativeList<Vector4> angularVelocitiesNative;
        public NativeList<Vector4> torquesNative;

        public NativeList<Vector4> intrinsicBendNative;
        public NativeList<Vector4> intrinsicBendKsNative;

        public NativeList<Vector4> posCorrectionNative;
        public NativeList<Quaternion> rotCorrectionNative;

        public NativeList<float> quatMassesInvNative;
        public NativeList<float> restLengthsNative;

        public NativeList<Vector3> stretchComplianceNative;
        public NativeList<Vector3> bendAndTorsionComplianceNative;
        public NativeList<float> lambdasNative;


        public Quaternion* orientationsNativePtr;
        public Quaternion* predictedOrientationsNativePtr;
        public Quaternion* prevOrientationsNativePtr;
        public Vector4* angularVelocitiesNativePtr;
        public Vector4* torquesNativePtr;
        public Vector4* intrinsicBendNativePtr;
        public Vector4* intrinsicBendKsNativePtr;

        public Vector3* stretchComplianceNativePtr;
        public Vector3* bendAndTorsionComplianceNativePtr;
        public float* lambdasNativePtr;

        public Vector4* posCorrectionNativePtr;
        public Quaternion* rotCorrectionNativePtr;

        public float* quatMassesInvNativePtr;
        public float* restLengthsNativePtr;

        

        public void Awake()
        {
            GetComponentInParent<CatheterGroup>().Configure();

            InitRuntime();
        }

        public void Start()
        {

            Body body = GetComponent<Body>();
            ElasticRod rod = GetComponent<ElasticRod>();
          //  if (solverType != ElasticRodSolverType.ITERATIVE)
            {
                rodPtr = DefKitDirectElasticRodSystem.InitDirectElasticRod(
                   size,
                   body.positionsNativePtr,
                   rod.orientationsNativePtr,
                   radius,
                   restLengthsNativePtr,
                   youngModulus,
                   torsionModulus);
            }

        }

        public virtual void InitRuntime()
        {
            InitNativeArrays();

            ResizeNativeArrays(this.size);
            Quaternion worldRot = transform.rotation;
            for (int i = 0; i < this.size; i++)
            {
                Quaternion q = worldRot * this.orientations[i];
                this.orientations[i] = q;
                this.orientationsNative[i] = q;
                this.predictedOrientationsNative[i] = q;
                this.prevOrientationsNative[i] = q;
                this.quatMassesInvNative[i] = this.quatMassesInv[i];
                this.restLengthsNative[i] = this.restLenghts[i];

                this.intrinsicBendNative[i] = this.intrinsicBend[i];
                this.intrinsicBendKsNative[i] = this.intrinsicBendKs[i];

                this.posCorrectionNative[i] = Vector4.zero;
                this.rotCorrectionNative[i] = new Quaternion(0,0,0,0);

                this.angularVelocitiesNative[i] = new Vector4();
                this.torquesNative[i] = new Vector4();

                this.stretchComplianceNative[i] = new Vector3();
                this.bendAndTorsionComplianceNative[i] = new Vector3();
                for (int j = 0; j < 6; j++)
                    this.lambdasNative[i * 6 + j] = 0;


            }
        }

        public virtual void InitManagedArrays(int count)
        {
            this.size = count;
            this.restOrientations = new Quaternion[count];
            this.orientations = new Quaternion[count];
            
            this.intrinsicBend = new Vector4[count];
            this.intrinsicBendKs = new Vector4[count];

            this.quatMassesInv = new float[count];
            this.restLenghts = new float[count];

        }

        public virtual void InitNativeArrays()
        {

            this.orientationsNative = new NativeList<Quaternion>(this.size, Allocator.Persistent);
            this.predictedOrientationsNative = new NativeList<Quaternion>(this.size, Allocator.Persistent);
            this.prevOrientationsNative = new NativeList<Quaternion>(this.size, Allocator.Persistent);
            this.angularVelocitiesNative = new NativeList<Vector4>(this.size, Allocator.Persistent);
            this.torquesNative = new NativeList<Vector4>(this.size, Allocator.Persistent);

            this.intrinsicBendNative = new NativeList<Vector4>(this.size, Allocator.Persistent);
            this.intrinsicBendKsNative = new NativeList<Vector4>(this.size, Allocator.Persistent);

            this.quatMassesInvNative = new NativeList<float>(this.size, Allocator.Persistent);
            this.restLengthsNative = new NativeList<float>(this.size, Allocator.Persistent);

            this.posCorrectionNative = new NativeList<Vector4>(this.size, Allocator.Persistent);
            this.rotCorrectionNative = new NativeList<Quaternion>(this.size, Allocator.Persistent);

            this.stretchComplianceNative = new NativeList<Vector3>(this.size, Allocator.Persistent);
            this.bendAndTorsionComplianceNative = new NativeList<Vector3>(this.size, Allocator.Persistent);
            this.lambdasNative = new NativeList<float>(this.size * 6, Allocator.Persistent);

            UpdatePointersToNativeArrays();
        }

        public void UpdatePointersToNativeArrays()
        {
            this.orientationsNativePtr = (Quaternion*) this.orientationsNative.GetUnsafePtr<Quaternion>();
            this.predictedOrientationsNativePtr = (Quaternion*)this.predictedOrientationsNative.GetUnsafePtr<Quaternion>();
            this.prevOrientationsNativePtr = (Quaternion*)this.prevOrientationsNative.GetUnsafePtr<Quaternion>();
            this.angularVelocitiesNativePtr = (Vector4*)this.angularVelocitiesNative.GetUnsafePtr<Vector4>();

            this.intrinsicBendNativePtr = (Vector4*)this.intrinsicBendNative.GetUnsafePtr<Vector4>();
            this.intrinsicBendKsNativePtr = (Vector4*)this.intrinsicBendKsNative.GetUnsafePtr<Vector4>();

            this.stretchComplianceNativePtr = (Vector3*)this.stretchComplianceNative.GetUnsafePtr<Vector3>();
            this.bendAndTorsionComplianceNativePtr = (Vector3*)this.bendAndTorsionComplianceNative.GetUnsafePtr<Vector3>();
            
            this.quatMassesInvNativePtr = (float*)this.quatMassesInvNative.GetUnsafePtr<float>();
            this.restLengthsNativePtr = (float*)this.restLengthsNative.GetUnsafePtr<float>();
            this.lambdasNativePtr = (float*)this.lambdasNative.GetUnsafePtr<float>();


            this.posCorrectionNativePtr = (Vector4*)this.posCorrectionNative.GetUnsafePtr<Vector4>();
            this.rotCorrectionNativePtr  = (Quaternion*)this.rotCorrectionNative.GetUnsafePtr<Quaternion>();
            this.torquesNativePtr = (Vector4*)this.torquesNative.GetUnsafePtr<Vector4>();
        }

        public void ResizeNativeArrays(int newLength)
        {
            this.orientationsNative.ResizeUninitialized(newLength);
            this.predictedOrientationsNative.ResizeUninitialized(newLength);
            this.prevOrientationsNative.ResizeUninitialized(newLength);
            this.angularVelocitiesNative.ResizeUninitialized(newLength);
            this.torquesNative.ResizeUninitialized(newLength);
            this.intrinsicBendNative.ResizeUninitialized(newLength);
            this.intrinsicBendKsNative.ResizeUninitialized(newLength);
            this.quatMassesInvNative.ResizeUninitialized(newLength);
            this.restLengthsNative.ResizeUninitialized(newLength);
            this.posCorrectionNative.ResizeUninitialized(newLength);
            this.rotCorrectionNative.ResizeUninitialized(newLength);
            this.stretchComplianceNative.ResizeUninitialized(newLength);
            this.bendAndTorsionComplianceNative.ResizeUninitialized(newLength);
            this.lambdasNative.ResizeUninitialized(newLength * 6);

            UpdatePointersToNativeArrays();
        }

        public void ResetSimulation()
        {
            this.orientationsNative.CopyFromNBC(this.orientations);
           
            for (int i = 0; i < this.size; i++)
            {
                this.angularVelocitiesNative[i] = default(Vector4);
            }
        }


        public virtual void DisposeArrays()
        {
            this.orientationsNative.Dispose();
            this.predictedOrientationsNative.Dispose();
            this.prevOrientationsNative.Dispose();
            this.angularVelocitiesNative.Dispose();
            this.torquesNative.Dispose();
            this.intrinsicBendNative.Dispose();

            this.intrinsicBendKsNative.Dispose();
            this.quatMassesInvNative.Dispose();
            this.restLengthsNative.Dispose();

            this.posCorrectionNative.Dispose();
            this.rotCorrectionNative.Dispose();

            this.stretchComplianceNative.Dispose();
            this.bendAndTorsionComplianceNative.Dispose();
            this.lambdasNative.Dispose();

            this.orientationsNativePtr = null;
            this.predictedOrientationsNativePtr = null;
            this.angularVelocitiesNativePtr = null;
            this.intrinsicBendNativePtr = null;

            this.posCorrectionNativePtr = null;
            this.rotCorrectionNativePtr = null;

            this.intrinsicBendKsNativePtr = null;
            this.quatMassesInvNativePtr = null;
            this.restLengthsNativePtr = null;
            this.torquesNativePtr = null;
            this.prevOrientationsNativePtr = null;

            this.stretchComplianceNativePtr = null;
            this.bendAndTorsionComplianceNativePtr = null;

            this.lambdasNativePtr = null;
        }



        public void SetRestLength(float restLength)
        {
            this.restLength = restLength;
        }

        void OnDestroy()
        {
            DisposeArrays();
        }
    }
}