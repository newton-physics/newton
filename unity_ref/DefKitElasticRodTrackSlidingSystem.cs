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

using Unity.Jobs;
using Unity.Collections;
using Unity.Mathematics;
using Unity.Burst;
using System;

namespace DefKit.ElasticRods
{
    /// <summary>
    /// Solver system which constraints particles to move along a track
    /// </summary>
    [Obsolete("This system is deprecated. It was merged into DefKitRodContolSystem. Please remove this component from the solver")]
    public class DefKitElasticRodTrackSlidingSystem : DefKitSolverSystem
    {
        public Body rodBody;
        public int ignoreTipCount = 0;

        public float stiffness = 1;
        
        public Transform trackStart;
        public Transform trackEnd;

        [BurstCompile]
        public struct RodSlidingJob : IJob
        {
            public float kS;
            public float4 start;
            public float4 end;
            public NativeArray<Vector4> positions;


            public int startId;
            public int endId;

            public void Execute()
            {

                for (int e = startId; e < endId; e++)
                {
                    float4 x = positions[e];
                    float t;
                    float4 p;

                    GeoUtils.ClosestPointOnEdge(x, start, end, out p, out t);

                    float4 n = p - x;
                    float lengthSq = math.lengthsq(n);

                    if (lengthSq > 0.000000001f && t >0 && t < 1)
                    {
                       // Debug.DrawLine(p.xyz, x.xyz, Color.green);
                       // n = n / C;
                        positions[e] += (Vector4)n * kS;
                    }

                }

            }
        }

        private void Start()
        {
            
        }


        //public override void OnConstraintsIterationStart(int subStepNum, int maxSubSteps)
        public override void OnPostConstraintsSolve(int subStepNum, int maxSubSteps)
        {

            if (rodBody != null && rodBody.isActiveAndEnabled)
            {

                var rodSlidingJob = new RodSlidingJob()
                {
                    positions = rodBody.predictedPositionsNative,
                    start = new float4(trackStart.transform.position, 0),
                    end = new float4(trackEnd.transform.position, 0),
                    startId = 0,
                    endId = rodBody.count - ignoreTipCount,
                    kS = stiffness
                };
                rodSlidingJob.Run();
            }
        }

    }
}