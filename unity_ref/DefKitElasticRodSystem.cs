/*---------------------------------------------------------------------
* Copyright(C) 2017-2020 KRN Labs 
* Dr Przemyslaw Korzeniowski <korzenio@gmail.com>
* All rights reserved.
*
* This file is part of DefKit.
* It can not be copied and/or distributed 
* without the express permission of Dr Przemyslaw Korzeniowski
* -------------------------------------------------------------------*/

using UnityEngine;
using System.Runtime.InteropServices;

namespace DefKit.ElasticRods
{

    /// <summary>
    /// Solver system which handles all the deformations of elastic rods
    /// </summary>
    public unsafe class DefKitElasticRodSystem : DefKitSolverSystem
    {
#if UNITY_IPHONE  && !UNITY_EDITOR
        [DllImport("__Internal")]
#else
        [DllImport("DefKit")]
#endif
        public static extern void PredictRotationsPBD(float dt, float damping, int pointsCount, Quaternion* rotPtr, Quaternion* predRotPtr, Vector4* angVelPtr, Vector4* torques, float* quatInvMasses);


#if UNITY_IPHONE  && !UNITY_EDITOR
        [DllImport("__Internal")]
#else
        [DllImport("DefKit")]
#endif
        public static extern void PredictRotationsPBD2(float dt, float damping, int pointsCount, Quaternion* rotPtr, Quaternion* predRotPtr, Vector4* angVelPtr, Vector4* torques, ref Vector4 inertiaTensor);


#if UNITY_IPHONE  && !UNITY_EDITOR
        [DllImport("__Internal")]
#else
        [DllImport("DefKit")]
#endif
        public static extern void IntegrateRotationsPBD(float dt, int pointsCount, Quaternion* rotPtr, Quaternion* predRotPtr, Quaternion* prevRotPtr, Vector4* angVelPtr, float* quatInvMasses);


#if UNITY_IPHONE  && !UNITY_EDITOR
        [DllImport("__Internal")]
#else
        [DllImport("DefKitAdv")]
#endif
        public static extern void ProjectElasticRodConstraints(int pointsCount, Vector4* positions, Quaternion* orientations, float* invMasses, float* quatInvMasses, Vector4* intrinsicBend, Vector4* intrinsicBendKs, float* restLengths, float stretchAndShearKs, float bendAndTwistKs);


        public float rotDamping = 0.001f;

        public ElasticRod[] rods;

        public ElasticRodConstraints[] rodSimCnstrs;

        private void Awake()
        {
            rods = FindObjectsOfType<ElasticRod>();

            rodSimCnstrs = FindObjectsOfType<ElasticRodConstraints>();
        }

        public void Start()
        {

        }

        public override void OnSubStepStart(float dt, int subStepNum, int maxSubSteps)
        {
            foreach (ElasticRod rod in rods)
            {
                PredictRotationsPBD(dt, rotDamping, rod.size, rod.orientationsNativePtr, rod.predictedOrientationsNativePtr, rod.angularVelocitiesNativePtr, rod.torquesNativePtr, rod.quatMassesInvNativePtr);
            }
        }


        public override void OnSubStepEnd(float dt, int subStepNum, int maxSubSteps)
        {
            foreach (ElasticRod rod in rods)
            {
                IntegrateRotationsPBD(dt, rod.size, rod.orientationsNativePtr, rod.predictedOrientationsNativePtr, rod.prevOrientationsNativePtr, rod.angularVelocitiesNativePtr, rod.quatMassesInvNativePtr);
            }

        }


        public unsafe override void OnConstraintsIterationStart(int subStepNum, int maxSubSteps)
        {
            for (int i = 0; i < rodSimCnstrs.Length; i++)
            {

                ElasticRod rod = rodSimCnstrs[i].elasticRod;
                Body body = rodSimCnstrs[i].body;

                for (int j = 0; j < rodSimCnstrs[i].constraintsIterations; j++)
                {
                    ProjectElasticRodConstraints(rod.size, body.predictedPositionsNativePtr, rod.predictedOrientationsNativePtr, body.massesInvNativePtr, rod.quatMassesInvNativePtr, rod.intrinsicBendNativePtr, rod.intrinsicBendKsNativePtr, rod.restLengthsNativePtr, rodSimCnstrs[i].stretchAndShearKs, rodSimCnstrs[i].bendAndTwistKs);
                }

            }
        }
    }
}
