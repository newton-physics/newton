/*---------------------------------------------------------------------
* Copyright(C) 2017-2018 Dr Przemyslaw Korzeniowski <korzenio@gmail.com>
* All rights reserved.
*
* This file is part of DefKit.
* It can not be copied and/or distributed 
* without the express permission of Dr Przemyslaw Korzeniowski
* -------------------------------------------------------------------*/

using System.Collections;
using System.Collections.Generic;
using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using System.Runtime.InteropServices;
using System;
using UnityEngine.Profiling;
using Unity.Collections.LowLevel.Unsafe;

namespace DefKit.ElasticRods
{
    public unsafe class DefKitDirectElasticRodSystem : DefKitSolverSystem
    {

        #region NATIVE
// #if UNITY_IPHONE && !UNITY_EDITOR
//         [DllImport("__Internal")]
// #else
//         [DllImport("DefKitAdv")]
// #endif
//         public unsafe static extern int InitAdv_native(string serial);

#if UNITY_IPHONE && !UNITY_EDITOR
        [DllImport("__Internal")]
#else
        [DllImport("DefKitAdv")]
#endif
        public unsafe static extern void ActivateLicense_native(string serialNumber);

#if UNITY_IPHONE && !UNITY_EDITOR
        [DllImport("__Internal")]
#else
        [DllImport("DefKitAdv")]
#endif
        public unsafe static extern void DectivateLicense_native(string serialNumber);

#if UNITY_IPHONE && !UNITY_EDITOR
        [DllImport("__Internal")]
#else
        [DllImport("DefKit")]
#endif
        public static extern void SetDebugFunction_native(IntPtr fp);


#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#else
        [DllImport("DefKit")]
#endif

        public unsafe static extern void PredictRotationsPBD(float dt, float damping, int pointsCount, Quaternion* rotPtr, Quaternion* predRotPtr, Vector4* angVelPtr, Vector4* torques, float* quatInvMasses);


#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#else
        [DllImport("DefKit")]
#endif
        public unsafe static extern void PredictRotationsPBD2(float dt, float damping, int pointsCount, Quaternion* rotPtr, Quaternion* predRotPtr, Vector4* angVelPtr, Vector4* torques, ref Vector4 inertiaTensor);


#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#else
        [DllImport("DefKit")]
#endif
        public unsafe static extern void IntegrateRotationsPBD(float dt, int pointsCount, Quaternion* rotPtr, Quaternion* predRotPtr, Quaternion* prevRotPtr, Vector4* angVelPtr, float* quatInvMasses);


#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#else 
        [DllImport("DefKitAdv")]
#endif
        public static extern void ProjectElasticRodConstraints(int pointsCount, Vector4* positions, Quaternion* orientations, float* invMasses, float* quatInvMasses, Vector4* intrinsicBend, Vector4* intrinsicBendKs, float* restLengths, float stretchAndShearKs, float bendAndTwistKs);


#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#elif UNITY_ANDROID && !UNITY_EDITOR
        [DllImport("DefKitAdv")]
#else
        [DllImport("DefKitAdv")]
#endif
        public unsafe static extern IntPtr InitDirectElasticRod(int edgesCount, Vector4* positions, Quaternion* orientations, float radius, float* restLengths, float youngModulus, float torsionModulus);


#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#elif UNITY_ANDROID && !UNITY_EDITOR
        [DllImport("DefKitAdv")]
#else
        [DllImport("DefKitAdv")]
#endif
        public unsafe static extern void PrepareDirectElasticRodConstraints(IntPtr rod, int pointsCount, float dt, Vector4* bendStiffness, Vector4* restDarboux, float* restLengths, float youngModulusMult, float torsionModulusMult);



#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#elif UNITY_ANDROID && !UNITY_EDITOR
        [DllImport("DefKitAdv")]
#else
        [DllImport("DefKitAdv")]
#endif
        public unsafe static extern void UpdateConstraints_DirectElasticRodConstraintsBanded(IntPtr rod, int pointsCount, Vector4* positions, Quaternion* orientations, float* invMasses);

#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#elif UNITY_ANDROID && !UNITY_EDITOR
        [DllImport("DefKitAdv")]
#else
        [DllImport("DefKitAdv")]
#endif
        public unsafe static extern void ComputeJacobians_DirectElasticRodConstraintsBanded(IntPtr rod, int startId, int pointsCount, Vector4* positions, Quaternion* orientations, float* invMasses);


        
#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#elif UNITY_ANDROID && !UNITY_EDITOR
        [DllImport("DefKitAdv")]
#else
        [DllImport("DefKitAdv")]
#endif
        public unsafe static extern void AssembleJMJT_DirectElasticRodConstraintsBanded(IntPtr rod, int startId, int pointsCount, Vector4* positions, Quaternion* orientations, float* invMasses);



#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#elif UNITY_ANDROID && !UNITY_EDITOR
        [DllImport("DefKitAdv")]
#else
        [DllImport("DefKitAdv")]
#endif
        public unsafe static extern void ProjectJMJT_DirectElasticRodConstraintsBanded(IntPtr rod, int pointsCount, Vector4* positions, Quaternion* orientations, float* invMasses, Vector4* posCorr, Quaternion* rotCorr);


#if UNITY_IPHONE && !UNITY_EDITOR
                [DllImport("__Internal")]
#elif UNITY_ANDROID && !UNITY_EDITOR
        [DllImport("DefKitAdv")]
#else
        [DllImport("DefKitAdv")]
#endif
        public unsafe static extern void DestroyDirectElasticRod(IntPtr rod);


        #endregion

        #region JOBS
        public unsafe struct ProjectElasticRodConstraintsNativeJob : IJob
        {
            public int pointsCount;

            public float stretchAndShearKs;

            public float bendAndTwistKs;


            [NativeDisableUnsafePtrRestriction]
            public Vector4* positionsPtr;

            [NativeDisableUnsafePtrRestriction]
            public Quaternion* orientationsPtr;


            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float* restLengthsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float* massesInvPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float* quatMassesInvPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Vector4* intrinsicBendPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Vector4* intrinsicBendKsPtr;



            public void Execute()
            {
                ProjectElasticRodConstraints(pointsCount, positionsPtr, orientationsPtr, massesInvPtr, quatMassesInvPtr, intrinsicBendPtr, intrinsicBendKsPtr, restLengthsPtr, stretchAndShearKs, bendAndTwistKs);
            }
        }

     

        public unsafe struct UpdateConstraintsDirectElasticRodConstraintsNativeJob : IJob
        {

            public int pointsCount;

            [NativeDisableUnsafePtrRestriction]
            public IntPtr rodPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Vector4* positionsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Quaternion* orientationsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float* massesInvPtr;


            public void Execute()
            {
                UpdateConstraints_DirectElasticRodConstraintsBanded(rodPtr, pointsCount, positionsPtr, orientationsPtr, massesInvPtr);
            }

        }

       


        public unsafe struct AssembleConstraintsDirectElasticRodConstraintsNativeJob : IJob
        {

            public int cnstrsCount;

            [NativeDisableUnsafePtrRestriction]
            public IntPtr rodPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Vector4* positionsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Quaternion* orientationsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float* massesInvPtr;


            public void Execute()
            {
                AssembleJMJT_DirectElasticRodConstraintsBanded(rodPtr, 0, cnstrsCount, positionsPtr, orientationsPtr, massesInvPtr);
            }

        }

        public unsafe struct ComputeJacobiansDirectElasticRodConstraintsNativeJobBatch : IJobParallelForBatch
        {
            [NativeDisableUnsafePtrRestriction]
            public IntPtr rodPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Vector4* positionsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Quaternion* orientationsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float* massesInvPtr;

            
            public void Execute(int startId, int count)
            {
                ComputeJacobians_DirectElasticRodConstraintsBanded(rodPtr, startId, count, positionsPtr, orientationsPtr, massesInvPtr);
            }

        }

        public unsafe struct AssembleDirectElasticRodConstraintsNativeJobBatch : IJobParallelForBatch
        {
            [NativeDisableUnsafePtrRestriction]
            public IntPtr rodPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Vector4* positionsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public Quaternion* orientationsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float* massesInvPtr;


            public void Execute(int startId, int count)
            {
                AssembleJMJT_DirectElasticRodConstraintsBanded(rodPtr, startId, count, positionsPtr, orientationsPtr, massesInvPtr);
            }

        }

       
        public unsafe struct SolveDirectElasticRodConstraintsNativeJob: IJob
        {

            public int pointsCount;

            [NativeDisableUnsafePtrRestriction]
            public IntPtr rodPtr;

            [NativeDisableUnsafePtrRestriction]
            public Vector4* positionsPtr;

            [NativeDisableUnsafePtrRestriction]
            public Quaternion* orientationsPtr;

            [ReadOnly]
            [NativeDisableUnsafePtrRestriction]
            public float* massesInvPtr;


            [NativeDisableUnsafePtrRestriction]
            public Vector4* posCorr;

            [NativeDisableUnsafePtrRestriction]
            public Quaternion* rotCorr;


            public void Execute()
            {
                ProjectJMJT_DirectElasticRodConstraintsBanded(rodPtr, pointsCount, positionsPtr, orientationsPtr, massesInvPtr, posCorr, rotCorr);
            }

        }
        #endregion


        public int batchSize = 64;

        public float subStepDt;

        public float rotDamping = 0.001f;

        public ElasticRod[] rods;
        public ElasticRodConstraints[] rodSimParams;

        private NativeArray<JobHandle> jobHandles;

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void DebugDelegate(string str);

        public SolverStage lambdasReset = SolverStage.ON_SUBSTEP_START;

        [AOT.MonoPInvokeCallback(typeof(DebugDelegate))]
        static void NativeDebugCallBack(string str)
        {
            Debug.Log("<color=blue>DefKitNative</color> " + str);
        }


        private void Awake()
        {
            DebugDelegate callback_delegate = new DebugDelegate(NativeDebugCallBack);

            // Convert callback_delegate into a function pointer that can be used in unmanaged code.
            IntPtr intptr_delegate = Marshal.GetFunctionPointerForDelegate(callback_delegate);

            // Call the API passing along the function pointer.
            SetDebugFunction_native(intptr_delegate);

         //   int res = InitAdv_native("serial");
          //  Debug.Log("InitAdv: " + res);


        }


        // Use this for initialization
        void Start()
        {
            rods = FindObjectsOfType<ElasticRod>();
            rodSimParams = FindObjectsOfType<ElasticRodConstraints>();
            jobHandles = new NativeArray<JobHandle>(rods.Length, Allocator.Persistent);

        }

        private void Update()
        {
            int rodsPrevCount = rods.Length;

            rods = FindObjectsOfType<ElasticRod>();
            rodSimParams = FindObjectsOfType<ElasticRodConstraints>();
            if (rodsPrevCount != rods.Length)
            {
                if (jobHandles.IsCreated)
                    jobHandles.Dispose();

                jobHandles = new NativeArray<JobHandle>(rods.Length, Allocator.Persistent);

            }
        }

        [ContextMenu("Activate Licence")]
        public void ActivateSerialNumber()
        {
           // ActivateLicense_native(serialNo);
        }

        [ContextMenu("Dectivate Licence")]
        public void DectivateSerialNumber()
        {
           // DectivateLicense_native(serialNo);
        }


        public override void OnSolverStart(float dt)
        {


            foreach (ElasticRod rod in rods)
            {
                //if (rod.isActiveAndEnabled)
                {

                    if (rod.solverType != ElasticRod.ElasticRodSolverType.ITERATIVE && lambdasReset == SolverStage.ON_SOLVER_START)
                    {
                        Profiler.BeginSample("PrepareDirectElasticRodConstraints");
                        PrepareDirectElasticRodConstraints(rod.rodPtr, rod.size - 1, dt, rod.intrinsicBendKsNativePtr, rod.intrinsicBendNativePtr, rod.restLengthsNativePtr, rod.youngModulus, rod.torsionModulus);
                        Profiler.EndSample();
                    }
                }
            }
            

        }

        public override void OnSubStepStart(float dt, int subStepNum, int maxSubSteps)
        {
            subStepDt = dt;
            foreach (ElasticRod rod in rods)
            {
                Profiler.BeginSample("PredictRotationsPBD");
                PredictRotationsPBD(dt, rotDamping, rod.size, rod.orientationsNativePtr, rod.predictedOrientationsNativePtr, rod.angularVelocitiesNativePtr, rod.torquesNativePtr, rod.quatMassesInvNativePtr);
                Profiler.EndSample();

                //if (rod.isActiveAndEnabled)
                {
                    if (rod.solverType != ElasticRod.ElasticRodSolverType.ITERATIVE && lambdasReset == SolverStage.ON_SUBSTEP_START)
                    {
                        Profiler.BeginSample("PrepareDirectElasticRodConstraints");
                        PrepareDirectElasticRodConstraints(rod.rodPtr, rod.size - 1, dt, rod.intrinsicBendKsNativePtr, rod.intrinsicBendNativePtr, rod.restLengthsNativePtr, rod.youngModulus, rod.torsionModulus);
                        Profiler.EndSample();
                    }
                }
            }
            
        }


        public override void OnSubStepEnd(float dt, int subStepNum, int maxSubSteps)
        {
            Profiler.BeginSample("IntegrateRotationsPBD");
            foreach (ElasticRod rod in rods)
            {
                //if (rod.isActiveAndEnabled)
                {
                    IntegrateRotationsPBD(dt, rod.size, rod.orientationsNativePtr, rod.predictedOrientationsNativePtr, rod.prevOrientationsNativePtr, rod.angularVelocitiesNativePtr, rod.quatMassesInvNativePtr);
                }
            }
            Profiler.EndSample();
        }

        public override void OnPreConstraintsSolve(int subStepNum, int maxSubSteps)
        {
            for (int i = 0; i < rodSimParams.Length; i++)
            {

                ElasticRod rod = rodSimParams[i].elasticRod;
                Body body = rodSimParams[i].body;

                if (rod.solverType != ElasticRod.ElasticRodSolverType.ITERATIVE && lambdasReset == SolverStage.ON_CONSTRAINTS_START)
                {
                        Profiler.BeginSample("PrepareDirectElasticRodConstraints");
                        PrepareDirectElasticRodConstraints(rod.rodPtr, rod.size - 1, subStepDt, rod.intrinsicBendKsNativePtr, rod.intrinsicBendNativePtr, rod.restLengthsNativePtr, rod.youngModulus, rod.torsionModulus);
                        Profiler.EndSample();
                }
                
            }


            ProjectRodConstraints(subStepNum, maxSubSteps, -1, -1);


        }

        private void ProjectRodConstraints(int subStepNum, int maxSubSteps, int constraintsIterNum, int contraintsIterMax)
        {
            for (int i = 0; i < rodSimParams.Length; i++)
            {

                ElasticRod rod = rodSimParams[i].elasticRod;
                Body body = rodSimParams[i].body;
                if (rod.solverType != ElasticRod.ElasticRodSolverType.ITERATIVE)
                {

                    var updateConstraintsJob = new UpdateConstraintsDirectElasticRodConstraintsNativeJob()
                    {
                        rodPtr = rod.rodPtr,
                        pointsCount = rod.size,
                        orientationsPtr = rod.predictedOrientationsNativePtr,
                        positionsPtr = body.predictedPositionsNativePtr,
                        massesInvPtr = body.massesInvNativePtr,
                    };
                    jobHandles[i] = updateConstraintsJob.Schedule(jobHandles[i]);


                    var jacobiansConstraintsJob = new ComputeJacobiansDirectElasticRodConstraintsNativeJobBatch()
                    {
                        rodPtr = rod.rodPtr,
                        orientationsPtr = rod.predictedOrientationsNativePtr,
                        positionsPtr = body.predictedPositionsNativePtr,
                        massesInvPtr = body.massesInvNativePtr,
                    };
                    jobHandles[i] = jacobiansConstraintsJob.ScheduleBatch(body.count - 1, batchSize, jobHandles[i]);

                    var assembleConstraintsJob = new AssembleDirectElasticRodConstraintsNativeJobBatch()
                    {
                        rodPtr = rod.rodPtr,
                        orientationsPtr = rod.predictedOrientationsNativePtr,
                        positionsPtr = body.predictedPositionsNativePtr,
                        massesInvPtr = body.massesInvNativePtr,
                    };
                    jobHandles[i] = assembleConstraintsJob.ScheduleBatch(body.count - 1, batchSize, jobHandles[i]);

                    var projectConstraintsJob = new SolveDirectElasticRodConstraintsNativeJob()
                    {
                        rodPtr = rod.rodPtr,
                        pointsCount = rod.size,
                        orientationsPtr = rod.predictedOrientationsNativePtr,
                        positionsPtr = body.predictedPositionsNativePtr,
                        massesInvPtr = body.massesInvNativePtr,
                        posCorr = rod.posCorrectionNativePtr,
                        rotCorr = rod.rotCorrectionNativePtr,
                    };
                    jobHandles[i] = projectConstraintsJob.Schedule(jobHandles[i]);
                    
                }
            }


            JobHandle.CompleteAll(jobHandles);
        }


        public unsafe override void OnConstraintsIterationStart(int subStepNum, int maxSubSteps, int constraintsIterNum, int contraintsIterMax)
        {
            Profiler.BeginSample("ProjectDirectElasticRodConstraints");
            for (int i = 0; i < rodSimParams.Length; i++)
            {

                ElasticRod rod = rodSimParams[i].elasticRod;
                Body body = rodSimParams[i].body;

                if (rod.solverType == ElasticRod.ElasticRodSolverType.ITERATIVE)
                {

                    for (int j = 0; j < rodSimParams[i].constraintsIterations; j++)
                    {
                        //  ProjectElasticRodConstraints(rod.size, body.predictedPositionsNativePtr, rod.predictedOrientationsNativePtr, body.massesInvNativePtr, rod.quatMassesInvNativePtr, rod.intrinsicBendNativePtr, rod.intrinsicBendKsNativePtr, rod.restLengthsNativePtr, rodSimParams[i].stretchAndShearKs, rodSimParams[i].bendAndTwistKs);
                        var projectConstraintsJob = new ProjectElasticRodConstraintsNativeJob()
                        {
                            pointsCount = rod.size,
                                
                            orientationsPtr = rod.predictedOrientationsNativePtr,
                            positionsPtr = body.predictedPositionsNativePtr,
                            massesInvPtr = body.massesInvNativePtr,
                            quatMassesInvPtr = rod.quatMassesInvNativePtr,
                            restLengthsPtr = rod.restLengthsNativePtr,
                            stretchAndShearKs = 1.0f,
                            bendAndTwistKs = 1.0f,
                            intrinsicBendKsPtr = rod.intrinsicBendKsNativePtr,
                            intrinsicBendPtr = rod.intrinsicBendNativePtr,
                        };
                      //  jobHandles[i] = projectConstraintsJob.Schedule(jobHandles[i]);
                        //projectConstraintsJob.Run();
                        }
                    
                }


            }
            JobHandle.CompleteAll(jobHandles);

            Profiler.EndSample();
        }

     

        //public void EnableDirectRod(bool direct)
        //{
        //    this.direct = direct;
        //}


        //public void EnableIterativeRod(bool iterative)
        //{
        //    this.iterative = iterative;
        //}


        private void OnDestroy()
        {
            if (jobHandles.IsCreated)
                jobHandles.Dispose();
        }
    }
}
