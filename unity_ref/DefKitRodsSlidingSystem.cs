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

using static Unity.Mathematics.math;

using DefKit.Instruments;
using System.Runtime.CompilerServices;

namespace DefKit.ElasticRods
{

    /// <summary>
    /// Solver system which constraints two rods together in a cocentric catheter/guidewire manner
    /// </summary>
    public class DefKitRodsSlidingSystem : DefKitSolverSystem
    {
        public AbstractVascularInput inputA;
        public AbstractVascularInput inputB;

        public ElasticRod rodA;
        public ElasticRod rodB;

        public float stiffness = 1;
        public float weightRatio = 0.5f;
       // public int weightRatioStartId = 0;
      //  public int startId = 0;
        public bool invMassSq = true;
        public bool hermiteInterpolation;
        public bool projectToNearest;
        public int expSearchRange = 2;

        public Color color;

        [Header("Info")]
        public float insertionDiff;


        private Body rodBodyA;
        private Body rodBodyB;

        private BodyExt rodBodyExtA;
        private BodyExt rodBodyExtB;

        [BurstCompile]
        public struct RodsSlidingJob : IJob
        {

            public bool stabilize;
            public bool invMassSq;

            public int pointsCountA;
            public int pointsCountB;

            public float kS;
            public float insertion;
            public float weight;

            public float restLengthA;
            public float restLengthB;


            public NativeArray<Vector4> positionsA;
            public NativeArray<Vector4> positionsB;

            public NativeArray<Vector4> predPositionsA;
            public NativeArray<Vector4> predPositionsB;


            public void Execute()
            {
                float wA = (1.0f - weight);
                float wB = weight;

                for (int i = 0; i < pointsCountA; i++)
                {
                    //float absolute = insertion + (i * restLengthA);
                    float absolute = (insertion / restLengthA + i);
                    float t = absolute % 1;

                    float b0 = 1.0f - t;
                    float b1 = t;

                    int j = (int)absolute;

                    if (j >= 0 && j < pointsCountB - 1)
                    {


                        //if(j > bilateralThresh)
                        //{
                        //    wA = 1.0f;
                        //    wB = 0.0f;
                        //}


                        float4 pB1 = predPositionsB[j];
                        float4 pB2 = predPositionsB[j + 1];

                        Vector4 v = lerp(pB1, pB2, t);
                        Vector4 dp = predPositionsA[i] - v;

                        float C = length(dp);

                        float s;
                        if (invMassSq)
                            s = wA + wB * b0 * b0 + wB * b1 * b1;
                        else
                            s = wA + wB * b0 + wB * b1;

                        s = C / s * kS;

                        //ease out at tip
                        if (j == pointsCountB - 2)
                        {
                            s *= (1.0f - t);

                        }

                            

                        if (s > 0.00000001f)
                        {
                            dp = normalize(dp);

                            predPositionsA[i] -= dp * s * wA;

                            predPositionsB[j] += dp * s * b0 * wB;
                            predPositionsB[j + 1] += dp * s * b1 * wB;


                            if (stabilize)
                            {
                                positionsA[i] -= dp * s * wA;

                                positionsB[j] += dp * s * b0 * wB;
                                positionsB[j + 1] += dp * s * b1 * wB;
                            }

                        }

                    }

                }

            }
        }

        
        [BurstCompile]
        public struct RodsSlidingVariableRestLengthsJob : IJob
        {

            public bool stabilize;
            public bool invMassSq;

            public int pointsCountA;
            public int pointsCountB;

            public float kS;
            public float insertion;
            public float weight;

            public Color color;

            public NativeArray<float> restLengthsA;
            public NativeArray<float> restLengthsB;


            public NativeArray<Vector4> positionsA;
            public NativeArray<Vector4> positionsB;

            public NativeArray<Vector4> predPositionsA;
            public NativeArray<Vector4> predPositionsB;

            public NativeArray<float> occlusionA;
            public NativeArray<float> occlusionB;

            public bool experimental;

            public void Execute()
            {
                float wA = (1.0f - weight);
                float wB = weight;

                float restLengthSumA = 0;
                for (int i = 0; i < pointsCountA; i++)
                {
                    //sum total rest length of A
                    float insA = insertion + restLengthSumA;

                    //first segment on B
                    int j = 0;
                    float prevRestLengthSumB = 0;
                    float restLengthSumB = restLengthsB[j];

                    //iterate through segments until insA in segment
                    while (restLengthSumB <= insA && j < pointsCountB - 2)
                    {
                        ++j;

                        prevRestLengthSumB = restLengthSumB;

                        if (experimental)
                        {
                            //int pA = Mathf.Max(j - 1, 0);
                            //int pB = j;
                            //int pC = Mathf.Min(j + 1, pointsCountB - 1);
                            //int pD = Mathf.Min(j + 2, pointsCountB - 1);

                            //float4 pB1 = predPositionsB[pB];
                            //float4 pB2 = predPositionsB[pC];
                            //float4 mB1 = 0.5f * (predPositionsB[pC] - predPositionsB[pA]);
                            //float4 mB2 = 0.5f * (predPositionsB[pD] - predPositionsB[pB]);

                            //Vector4 previousV = pB1;
                            //for (int k = 0; k < 10; k++)
                            //{
                            //    Vector4 v = HermiteInterpolate(pB1, pB2, mB1, mB2, k / 10);
                            //    restLengthSumB += Vector4.Distance(previousV, v);
                            //    previousV = v;
                            //}
                            //restLengthSumB += Vector4.Distance(previousV, pB2);
                            restLengthSumB += restLengthsB[j];
                        }
                        else
                            restLengthSumB += restLengthsB[j];
                    }
                    


                    //to start at zero
                    float a = insA - prevRestLengthSumB;
                    float b = restLengthSumB - prevRestLengthSumB;
                    float t = a / b;

                    restLengthSumA += restLengthsA[i];

               
                    if (t < 0 || t > 1)
                        continue;

                    if (j >= 0 && j < pointsCountB - 1)
                    {
                        int pA = Mathf.Max(j - 1, 0);
                        int pB = j;
                        int pC = Mathf.Min(j + 1, pointsCountB - 1);
                        int pD = Mathf.Min(j + 2, pointsCountB - 1);

                        float4 pB1 = predPositionsB[pB];
                        float4 pB2 = predPositionsB[pC];
                        float4 mB1 = 0.5f * (predPositionsB[pC] - predPositionsB[pA]);
                        float4 mB2 = 0.5f * (predPositionsB[pD] - predPositionsB[pB]);

                        //float4 pB1 = predPositionsB[j];
                        //float4 pB2 = predPositionsB[j + 1];

                        Vector4 v;
                        if (experimental)
                            v = HermiteInterpolate(pB1, pB2, mB1, mB2, t);
                        else
                            v = lerp(pB1, pB2, t);
                        Vector4 dp = predPositionsA[i] - v;

                        Vector4 segmentDirectionB = pB2 - pB1;
                        //if (i == 98)
                        //    Debug.Log("projection: " + (dp - Vector4.Project(dp, segmentDirectionB)));
                        dp -= 0.99f*Vector4.Project(dp, segmentDirectionB);

                        //Debug.DrawLine(positionsA[i], v, color);

                        float C = length(dp);

                        float b0 = 1.0f - t;
                        float b1 = t;

                        float s;
                        if (invMassSq)
                            s = wA + wB * b0 * b0 + wB * b1 * b1;
                        else
                            s = wA + wB * b0 + wB * b1;
                        s = C / s * kS;

                        //ease out at tip
                        if (j == pointsCountB - 2)
                        {
                            s *= (1.0f - t);
                            occlusionA[i] = t;
                        }
                        else
                        {
                            occlusionA[i] = 0;
                        }


                        if (s > 0.00000001f)
                        {
                            dp = normalize(dp);

                            predPositionsA[i] -= dp * s * wA;

                            predPositionsB[j] += dp * s * b0 * wB;
                            predPositionsB[j + 1] += dp * s * b1 * wB;

                            if (stabilize)
                            {
                                positionsA[i] -= dp * s * wA;

                                positionsB[j] += dp * s * b0 * wB;
                                positionsB[j + 1] += dp * s * b1 * wB;
                            }

                        }

                    }
                    else
                    {
                        occlusionA[i] = 1;
                    }
                }

            }
        }

        public static void ClosestPointOnRod(Vector4 pA, NativeArray<Vector4> positionsB, int countB, int startId, int endId, out Vector4 closestPoint, out float t, out int segId, out float closestDist, out bool outside)
        {
            closestDist = float.MaxValue;
            closestPoint = new Vector3();
            segId = -1;
            t = -1;
            outside = false;
            for (int i = Mathf.Max(0, startId); i < Mathf.Min(endId, countB - 1); i++)
            {
                float t2;
                Vector3 pointOnEdge;
                GeoUtils.ClosestPointOnEdge(pA, positionsB[i], positionsB[i + 1], out pointOnEdge, out t2);

                float dist = Vector3.Distance(pA, pointOnEdge);
                if (dist < closestDist)
                {
                    closestDist = dist;
                    closestPoint = pointOnEdge;
                    segId = i;
                    t = t2;
                    outside = (i == countB - 2 && t >= 1.0f) || (i == 0  && t <= 0.0f); 

                }
            }

        }


        [BurstCompile]
        public struct RodsSlidingJobNearestBruteForce : IJob
        {
            public float kS;

            public float insertion;

            public float weight;

            public float restLengthA;
            public float restLengthB;

            public int startId;

            public int searchRangeExp;

            public NativeArray<Vector4> positionsA;
            public NativeArray<Vector4> positionsB;

            public NativeArray<Vector4> predPositionsA;
            public NativeArray<Vector4> predPositionsB;

            public int pointsCountA; //guidewire
            public int pointsCountB; //catheter

            public bool stabilize;

            public void Execute()
            {

                float wA = (1.0f - weight);
                float wB = weight;

                for (int i = 0; i < pointsCountA; i++)
                {

                    Vector4 v;
                    float t;
                    float dist;
                    int segId;
                    //float absolute = insertion + (i * restLengthA);
                    float absolute = (insertion / restLengthA + i);
                    bool outside;

                  //  ClosestPointOnRod(predPositionsA[i], predPositionsB, pointsCountB, (int)absolute - 3, (int)absolute + 3, out v, out t, out segId, out dist, out outside);
                  //ClosestPointOnRod(predPositionsA[i], predPositionsB, pointsCountB, (int)absolute - searchRangeExp, (int)absolute + searchRangeExp, out v, out t, out segId, out dist, out outside);
                    ClosestPointOnRod(predPositionsA[i], predPositionsB, pointsCountB, 0, pointsCountB, out v, out t, out segId, out dist, out outside);


                    int j = segId;

                    if (j > 0 && j < pointsCountB - 1 && !outside)
                    //if (j > 0 && j < pointsCountB - 1)
                    {
                        Vector3 pB1 = predPositionsB[j];
                        Vector3 pB2 = predPositionsB[j + 1];

                        float b0 = 1.0f - t;
                        float b1 = t;

                        Vector4 dp = predPositionsA[i] - v;
                        float C = Vector3.Magnitude(dp);

                        float s = wA + wB * b0 * b0 + wB * b1 * b1;
                        s = C / s * kS;

                        //ease out at tip
                        if (j == pointsCountB - 2)
                            s *= (1.0f - t);

                        if (i >= startId && s > 0.0000001f)
                        {
                            dp = Vector3.Normalize(dp);

                            predPositionsA[i] -= dp * s * wA;

                            predPositionsB[j] += dp * s * b0 * wB;
                            predPositionsB[j + 1] += dp * s * b1 * wB;
                            
                            if(stabilize)
                            {
                                positionsA[i] -= dp * s * wA;

                                positionsB[j] += dp * s * b0 * wB;
                                positionsB[j + 1] += dp * s * b1 * wB;
                            }

                            // Debug.DrawLine(positionsA[i], v, Color.white);

                            //Debug.DrawRay(positionsA[i], -dp * s, Color.red);
                            //Debug.DrawRay(positionsB[j], dp * s * b0, Color.green);
                            //Debug.DrawRay(positionsB[j + 1], dp * s * b1, Color.green);
                        }

                    }
                }

            }
        }

        [BurstCompile]
        public struct RodsSlidingJobNearest : IJob
        {
            public float kS;

            public float insertion;

            public float weight;

            public int searchRange;

            public int startId;

            public int pointsCountA; //guidewire
            public int pointsCountB; //catheter


            public NativeArray<Vector4> positionsA;
            public NativeArray<Vector4> positionsB;

            public NativeArray<Vector4> predPositionsA;
            public NativeArray<Vector4> predPositionsB;

            //public NativeArray<byte> collisionIgnoresA;
            //public NativeArray<byte> collisionIgnoresB;

            public NativeArray<float> occlusionA;
            public NativeArray<float> occlusionB;


            [ReadOnly]
            public NativeArray<float> restLengthsA;

            [ReadOnly]
            public NativeArray<float> restLengthsB;

            public Color color;

            public bool stabilize;

            public void Execute()
            {

                float wA = (1.0f - weight);
                float wB = weight;

                float restLengthSumA = 0;
                for (int i = 0; i < pointsCountA; i++)
                {

                    //sum total rest length of A
                    float insA = insertion + restLengthSumA;

                    //first segment on B
                    int j = 0;
                    float prevRestLengthSumB = 0;
                    float restLengthSumB = restLengthsB[j];

                    //iterate through segments until insA in segment
                    while (restLengthSumB <= insA && j < pointsCountB - 2)
                    {
                        ++j;
                        prevRestLengthSumB = restLengthSumB;
                        restLengthSumB += restLengthsB[j];
                    }

                    restLengthSumA += restLengthsA[i];

                    Vector4 v;
                    float t;
                    float dist;
                    int segId;
                    bool outside;

                    //search for a closest point on other rod in the approximated range
                    ClosestPointOnRod(predPositionsA[i], predPositionsB, pointsCountB, j - searchRange, j + searchRange, out v, out t, out segId, out dist, out outside);

                    j = segId;


                    if (j > 0 && j < pointsCountB - 1 && !outside)
                    //if (j >= 0 && j < pointsCountB - 1)
                    {

                        Vector3 pB1 = predPositionsB[j];
                        Vector3 pB2 = predPositionsB[j + 1];

                        Vector4 dp = predPositionsA[i] - v;
                        float C = Vector3.Magnitude(dp);

                        float b0 = 1.0f - t;
                        float b1 = t;

                        float s = wA + wB * b0 * b0 + wB * b1 * b1;
                        s = C / s * kS;

                       // Debug.DrawLine(predPositionsA[i], v, color);

                        //ease out at tip
                        if (j == pointsCountB - 2)
                            s *= (1.0f - t);


                        //ease out at tip
                        if (j == pointsCountB - 2)
                        {
                            s *= (1.0f - t);
                          //  collisionIgnoresA[i] = 0;
                            occlusionA[i] = t;
                        }
                        else
                        {
                       //     collisionIgnoresA[i] = 1;
                            occlusionA[i] = 0;
                        }


                        if (i >= startId && s > 0.0000001f)
                        {
                            dp = Vector3.Normalize(dp);

                            predPositionsA[i] -= dp * s * wA;

                            predPositionsB[j] += dp * s * b0 * wB;
                            predPositionsB[j + 1] += dp * s * b1 * wB;

                            if (stabilize)
                            {
                                positionsA[i] -= dp * s * wA;

                                positionsB[j] += dp * s * b0 * wB;
                                positionsB[j + 1] += dp * s * b1 * wB;
                            }

                        }

                    }
                }

            }
        }

        private void Start()
        {
            rodBodyA = rodA.GetComponent<Body>();
            rodBodyB = rodB.GetComponent<Body>();

            rodBodyExtA = rodA.GetComponent<BodyExt>();
            rodBodyExtB = rodB.GetComponent<BodyExt>();
        }

        public override void OnSubStepStart(float dt, int subStepNum, int maxSubSteps)
        {
           // ProjectConstraints(subStepNum, maxSubSteps, true);
        }

        public override void OnStabilizationIterationStart(int subStepNum, int maxSubSteps)
        {
            ProjectConstraints(subStepNum, maxSubSteps, true);
        }

        public override void OnConstraintsIterationStart(int subStepNum, int maxSubSteps)
        {
            ProjectConstraints(subStepNum, maxSubSteps, false);
        }

        private void ProjectConstraints(int subStepNum, int maxSubSteps, bool stabilize)
        {
            //insertionDiff = rodBodyA.positionsNative[100].z - rodBodyB.positionsNative[100].z;
            //insertionDiff = inputA.insertion - inputB.insertion;

            float t = (float)subStepNum / (maxSubSteps - 1);
            if (maxSubSteps == 1)
                t = 0f;

            float insA = math.lerp(inputA.prevInsertion, inputA.insertion, t);
            float insB = math.lerp(inputB.prevInsertion, inputB.insertion, t);
            insertionDiff = insA - insB;

            if (projectToNearest)
            {

                var rodSlidingJob = new RodsSlidingJobNearestBruteForce()
                {
                    kS = stiffness,
                    stabilize = stabilize,
                    weight = weightRatio,
                    //startId = startId,
                    startId = 0,
                    insertion = insertionDiff,
                    searchRangeExp = expSearchRange,
                    pointsCountA = rodBodyA.count,
                    pointsCountB = rodBodyB.count,
                    restLengthA = rodA.restLength,
                    restLengthB = rodB.restLength,
                    positionsA = rodBodyA.positionsNative,
                    positionsB = rodBodyB.positionsNative,
                    predPositionsA = rodBodyA.predictedPositionsNative,
                    predPositionsB = rodBodyB.predictedPositionsNative,

                };
                rodSlidingJob.Run();

            //var rodSlidingJob = new RodsSlidingJobNearest()
            //{
            //    kS = stiffness,
            //    stabilize = stabilize,
            //    weight = weightRatio,
            //    //startId = startId,
            //    startId = 0,
            //    insertion = insertionDiff,
            //    searchRange = expSearchRange,
            //    pointsCountA = rodBodyA.count,
            //    pointsCountB = rodBodyB.count,
            //    restLengthsA = rodA.restLengthsNative,
            //    restLengthsB = rodB.restLengthsNative,
            //    positionsA = rodBodyA.positionsNative,
            //    positionsB = rodBodyB.positionsNative,
            //    predPositionsA = rodBodyA.predictedPositionsNative,
            //    predPositionsB = rodBodyB.predictedPositionsNative,
            //    occlusionA = rodBodyExtA.occlusionNative,
            //    occlusionB = rodBodyExtB.occlusionNative,
            //    color = color

            //};
            //rodSlidingJob.Run();

        }
            else
            {
                //var rodSlidingJob = new RodsSlidingJob()
                //{
                //    kS = stiffness,
                //    invMassSq = invMassSq,
                //    stabilize = stabilize,
                //    weight = weightRatio,
                //    bilateralThresh = weightRatioStartId,
                //    insertion = insertionDiff,
                //    pointsCountA = rodBodyA.count,
                //    pointsCountB = rodBodyB.count,
                //    restLengthA = rodA.restLength,
                //    restLengthB = rodB.restLength,
                //    positionsA = rodBodyA.positionsNative,
                //    positionsB = rodBodyB.positionsNative,
                //    predPositionsA = rodBodyA.predictedPositionsNative,
                //    predPositionsB = rodBodyB.predictedPositionsNative,
                //    collisionIgnoresA = rodBodyExtA.collisionsIgnoreNative,
                //    collisionIgnoresB = rodBodyExtB.collisionsIgnoreNative,
                //};

                var rodSlidingJob = new RodsSlidingVariableRestLengthsJob()
                {
                    kS = stiffness,
                    invMassSq = invMassSq,
                    stabilize = stabilize,
                    weight = weightRatio,
                    insertion = insertionDiff,
                    pointsCountA = rodBodyA.count,
                    pointsCountB = rodBodyB.count,
                    restLengthsA = rodA.restLengthsNative,
                    restLengthsB = rodB.restLengthsNative,
                    positionsA = rodBodyA.positionsNative,
                    positionsB = rodBodyB.positionsNative,
                    predPositionsA = rodBodyA.predictedPositionsNative,
                    predPositionsB = rodBodyB.predictedPositionsNative,
                    occlusionA = rodBodyExtA.occlusionNative,
                    occlusionB = rodBodyExtB.occlusionNative,
                    color = color,
                    experimental = hermiteInterpolation
                };
                rodSlidingJob.Run();
            }

            
        }

        public void SetWeightRatio(float weight)
        {
            this.weightRatio = weight;
        }

        public void SetStiffness(float stiffness)
        {
            this.stiffness = stiffness;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float4 HermiteInterpolate(float4 a, float4 b, float4 t1, float4 t2, float t)
        {
            // blending weights
            float w1 = 1.0f - 3 * t * t + 2 * t * t * t;
            float w2 = t * t * (3.0f - 2.0f * t);
            float w3 = t * t * t - 2 * t * t + t;
            float w4 = t * t * (t - 1.0f);

            // return weighted combination
            return a * w1 + b * w2 + t1 * w3 + t2 * w4;
        }

    }
}