using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;
using System.Runtime.CompilerServices;

namespace DefKit.ElasticRods
{
    /// <summary>
    /// Renders rope mesh using Hermite interpolation.
    /// </summary>
    public class RodMesher : MonoBehaviour
    {
        public Body m_body;
        public BodyExt m_bodyExt;

        public bool jobified;
        public float m_radius = 0.5f;
        public float m_radiusMult = 1.0f;
        public int m_resolution = 8;
        public int m_smoothing = 3;

        public float m_textureU = 1.0f;
        public float m_textureV = 1.0f;

        public int m_startId = 0;

        //-1 for full length
        public int m_pointsCount = -1;

        public AnimationCurve m_radiusScale;
        private Mesh m_mesh;

        private float[] m_radii;

        [SerializeField, HideInInspector]
        private Vector3[] m_vertices;

        [SerializeField, HideInInspector]
        private Vector3[] m_normals;

        [SerializeField, HideInInspector]
        private Vector2[] m_uvs;

        [SerializeField, HideInInspector]
        private Color[] m_colors;

        [SerializeField, HideInInspector]
        private int[] m_triangles;


        [SerializeField, HideInInspector]
        private NativeArray<Vector3> m_verticesNative;

        [SerializeField, HideInInspector]
        private NativeArray<Vector3> m_normalsNative;

        [SerializeField, HideInInspector]
        private NativeArray<Vector2> m_uvsNative;

        [SerializeField, HideInInspector]
        private NativeArray<Color> m_colorsNative;

        [SerializeField, HideInInspector]
        private NativeArray<int> m_trianglesNative;

        [SerializeField, HideInInspector]
        private NativeArray<float> m_radiiNative;


        [BurstCompile]
        struct UpdateRodMeshJob : IJob
        {

            public int startId;
            public int numPoints;
            public float radius;
            public float radiusMult;
            public int resolution;
            public int smoothing;

            public float m_textureU;
            public float m_textureV;

            public float4x4 tr;

            [ReadOnly]
            public NativeArray<float> radii;

            [ReadOnly]
            public NativeArray<Vector4> points;

            [ReadOnly]
            public NativeArray<Color> colors;

            [WriteOnly]
            public NativeArray<Vector3> verticesOut;

            [WriteOnly]
            public NativeArray<Vector3> normalsOut;

            [WriteOnly]
            public NativeArray<Color> colorsOut;

            [WriteOnly]
            public NativeArray<Vector2> uvsOut;


            public void Execute()
            {
                if (numPoints < 2)
                    return;

                Vector3 u, v;
                Vector3 w = Vector3.Normalize(points[1] - points[0]);

                BasisFromVector(w, out u, out v);

                Matrix4x4 frame = new Matrix4x4();
                frame.SetColumn(0, new Vector4(u.x, u.y, u.z, 0.0f));
                frame.SetColumn(1, new Vector4(v.x, v.y, v.z, 0.0f));
                frame.SetColumn(2, new Vector4(w.x, w.y, w.z, 0.0f));
                frame.SetColumn(3, new Vector4(0.0f, 0.0f, 0.0f, 1.0f));

                float uvStepH = m_textureU / (float)resolution;
                float uvStepV = m_textureV / ((float)numPoints * smoothing);

                int vId = 0;
                for (int i = startId; i < startId + numPoints - 1; ++i)
                {
                    Vector3 next = new Vector3();

                    if (i > 0)
                    {
                        if (i < startId + numPoints - 1)
                            next = Vector3.Normalize(points[i + 1] - points[i - 1]);
                        else
                            next = Vector3.Normalize(points[i] - points[i - 1]);
                    }
                   
                    int a = Mathf.Max(i - 1, 0);
                    int b = i;
                    int c = Mathf.Min(i + 1, startId + numPoints - 1);
                    int d = Mathf.Min(i + 2, startId + numPoints - 1);

                    Vector3 p1 = points[b];
                    Vector3 p2 = points[c];
                    Vector3 m1 = 0.5f * (points[c] - points[a]);
                    Vector3 m2 = 0.5f * (points[d] - points[b]);

                    float r1 = radii[b];
                    float r2 = radii[c];
                    float rm1 = 0.5f * (radii[c] - radii[a]);
                    float rm2 = 0.5f * (radii[d] - radii[b]);
                    // ensure last segment handled correctly
                    int segments = (i < startId + numPoints - 2) ? smoothing : smoothing + 1;

                    for (int s = 0; s < segments; ++s)
                    {
                        Vector4 pos = HermiteInterpolate(p1, p2, m1, m2, s / (float)smoothing);
                        Vector3 dir = Vector3.Normalize(HermiteTangent(p1, p2, m1, m2, s / (float)smoothing));
                        //float r = Mathf.Lerp(r1, r2, s / (float)smoothing);
                        float r = (HermiteInterpolate(r1, r2, rm1, rm2, s / (float)smoothing) + radius) * radiusMult;

                        //Vector3 cur = frame.GetAxis(2);
                        //Note intentional Vector4.w component drop
                        Vector3 cur = frame.GetColumn(2);
                        float angle = Mathf.Acos(Vector3.Dot(cur, dir));

                        // if parallel then don't need to do anything
                        if (Mathf.Abs(angle) > 0.001f)
                        {
                            //  Quaternion q = Quaternion.AngleAxis(angle, Vector3.Normalize(Vector3.Cross(cur, dir)));
                            //  frame = Matrix4x4.TRS(Vector3.zero, q, Vector3.one) * frame;
                            frame = RotationMatrix(angle, Vector3.Normalize(Vector3.Cross(cur, dir))) * frame;
                        }

                        for (int cc = 0; cc < resolution; ++cc)
                        {
                            float angle2 = 2.0f * Mathf.PI / (float)resolution;

                            // transform position and normal to world space
                            Vector4 vv = frame * new Vector4(Mathf.Cos(angle2 * cc), Mathf.Sin(angle2 * cc), 0.0f, 0.0f);
                            //verticesOut[vId] = tr.InverseTransformPoint(vv * radius + pos);

                            // verticesOut[vId] = tr.InverseTransformPoint(vv * m_radii[i - startId] + pos);
                            // normalsOut[vId] = tr.InverseTransformDirection(vv);


                            //verticesOut[vId] = vv * radii[i - startId] * radius + pos;
                            verticesOut[vId] = vv * r + pos;
                            //verticesOut[vId] = vv * radii[i * smoothing + s] * radius + pos;
                            normalsOut[vId] = vv;


                            uvsOut[vId] = new Vector2(cc * uvStepH, (i * segments + s) * uvStepV);
                            colorsOut[vId] = colors[i];
                            //uvsOut[vId] = new Vector2(s * uvStepH, i * uvStepV);
                            //UVs[j * numVertexColumns + i] = new Vector2(i * uvStepH, j * uvStepV);
                            vId++;
                        }

                    }

                }
            }
        }

        // Use this for initialization
        void Start()
        {
            if (m_body == null)
                m_body = GetComponent<Body>();

            m_mesh = new Mesh();
            m_mesh.name = "RopeMesh";

            if (m_pointsCount == -1)
                m_pointsCount = Mathf.Min(m_body.count - m_startId, m_body.count);

            m_radii = new float[m_pointsCount * m_smoothing];
            m_uvs = new Vector2[m_resolution * m_pointsCount * m_smoothing];
            m_colors = new Color[m_resolution * m_pointsCount * m_smoothing];

            for (int i = 0; i < m_pointsCount * m_smoothing; i++)
            {
                m_radii[i] =  m_radiusScale.Evaluate((float)i / (m_pointsCount * m_smoothing - 1));
            }

            InitRope(m_startId, m_pointsCount);

            m_verticesNative = new NativeArray<Vector3>(m_vertices, Allocator.Persistent);
            m_normalsNative = new NativeArray<Vector3>(m_normals, Allocator.Persistent);
            m_colorsNative = new NativeArray<Color>(m_colors, Allocator.Persistent);
            m_uvsNative = new NativeArray<Vector2>(m_uvs, Allocator.Persistent);
            m_trianglesNative = new NativeArray<int>(m_triangles, Allocator.Persistent);
            m_radiiNative = new NativeArray<float>(m_radii, Allocator.Persistent);

            GetComponent<MeshFilter>().sharedMesh = m_mesh;


        }

        // Update is called once per frame
        void LateUpdate()
        {
            if (jobified)
            {
                var updateRodMeshJob = new UpdateRodMeshJob()
                {
                    startId = m_startId,
                    numPoints = m_pointsCount,
                    radius = m_radius,
                    radiusMult = m_radiusMult,
                    resolution = m_resolution,
                    smoothing = m_smoothing,
                    m_textureU = m_textureU,
                    m_textureV = m_textureV,
                    tr = transform.localToWorldMatrix,
                    points = m_body.positionsNative,
                    colors = m_bodyExt.colorsNative,
                    verticesOut = m_verticesNative,
                    normalsOut = m_normalsNative,
                    colorsOut = m_colorsNative,
                    uvsOut = m_uvsNative,
                    //radii = m_radiiNative
                    radii = m_bodyExt.radiusNative
                };
                updateRodMeshJob.Run();

                m_verticesNative.CopyToFast(m_vertices, 0, m_vertices.Length);
                m_normalsNative.CopyToFast(m_normals, 0, m_normals.Length);
                m_uvsNative.CopyToFast(m_uvs, 0, m_uvs.Length);
                m_colorsNative.CopyToFast(m_colors, 0, m_colors.Length);

                m_mesh.vertices = m_vertices;
                m_mesh.normals = m_normals;
                m_mesh.uv = m_uvs;
                m_mesh.colors = m_colors;
            }
            else
            {
                //   DrawRope(transform, m_body.positions, m_bodyExt.colors, m_startId, m_pointsCount, ref m_vertices, ref m_normals, ref m_colors, ref m_uvs, m_radius, m_resolution, m_smoothing);
                DrawRope(transform, m_body.positionsNative, m_bodyExt.colorsNative, m_startId, m_pointsCount, ref m_vertices, ref m_normals, ref m_colors, ref m_uvs, m_radius * m_radiusMult, m_resolution, m_smoothing);

                m_mesh.vertices = m_vertices;
                m_mesh.normals = m_normals;
                m_mesh.uv = m_uvs;
                m_mesh.colors = m_colors;
            }



            m_mesh.RecalculateBounds();

            //InitRope(m_startId, m_pointsCount);
        }

        public void InitRope(int startId, int numPoints)
        {


            m_vertices = new Vector3[m_resolution * numPoints * m_smoothing];
            m_normals = new Vector3[m_resolution * numPoints * m_smoothing];
            m_uvs = new Vector2[m_resolution * numPoints * m_smoothing];
            m_colors = new Color[m_resolution * m_pointsCount * m_smoothing];
            m_triangles = new int[numPoints * m_resolution * 6 * m_smoothing];

            DrawRope(transform, m_body.positions, m_bodyExt.colors, startId, numPoints, ref m_vertices, ref m_normals, ref m_colors, ref m_uvs, m_radius * m_radiusMult, m_resolution, m_smoothing);

            UpdateRopeTriangles(startId, numPoints, ref m_triangles, m_radius * m_radiusMult, m_resolution, m_smoothing);

            m_mesh.vertices = m_vertices;
            m_mesh.normals = m_normals;
            m_mesh.uv = m_uvs;
            m_mesh.colors = m_colors;
            m_mesh.triangles = m_triangles;

            m_mesh.RecalculateBounds();
        }

        public void UpdateRopeTriangles(int startId, int numPoints, ref int[] trianglesOut, float radius, int resolution, int smoothing)
        {
            int startIndex = 0;



            List<int> triangles = new List<int>();
            for (int i = startId; i < startId + numPoints - 1; ++i)
            {
                int segments = (i < startId + numPoints - 2) ? m_smoothing : m_smoothing + 1;

                for (int s = 0; s < segments; ++s)
                {

                    // output triangles
                    if (startIndex != 0)
                    {
                        //    Debug.Log(startIndex);
                        for (int ii = 0; ii < m_resolution; ++ii)
                        {
                            int curIndex = startIndex + ii;
                            int nextIndex = startIndex + (ii + 1) % m_resolution;

                            triangles.Add(curIndex);
                            triangles.Add(curIndex - m_resolution);
                            triangles.Add(nextIndex - m_resolution);

                            triangles.Add(nextIndex - m_resolution);
                            triangles.Add(nextIndex);
                            triangles.Add(curIndex);
                        }
                    }

                    startIndex += m_resolution;
                }
            }

            trianglesOut = triangles.ToArray();
        }

        public void DrawRope(Transform tr, Vector4[] points, Color[] colors, int startId, int numPoints, ref Vector3[] verticesOut, ref Vector3[] normalsOut, ref Color[] colorsOut, ref Vector2[] uvsOut, float radius, int resolution, int smoothing)
        {

            if (numPoints < 2)
                return;

            Vector3 u, v;
            Vector3 w = Vector3.Normalize(points[1] - points[0]);

            BasisFromVector(w, out u, out v);

            Matrix4x4 frame = new Matrix4x4();
            frame.SetColumn(0, new Vector4(u.x, u.y, u.z, 0.0f));
            frame.SetColumn(1, new Vector4(v.x, v.y, v.z, 0.0f));
            frame.SetColumn(2, new Vector4(w.x, w.y, w.z, 0.0f));
            frame.SetColumn(3, new Vector4(0.0f, 0.0f, 0.0f, 1.0f));

            float uvStepH = m_textureU / (float)resolution;
            float uvStepV = m_textureV / ((float)numPoints * smoothing);

            int vId = 0;
            for (int i = startId; i < startId + numPoints - 1; ++i)
            {
                Vector3 next = new Vector3();

                if (i > 0)
                {
                    if (i < startId + numPoints - 1)
                        next = Vector3.Normalize(points[i + 1] - points[i - 1]);
                    else
                        next = Vector3.Normalize(points[i] - points[i - 1]);
                }

                int a = Mathf.Max(i - 1, 0);
                int b = i;
                int c = Mathf.Min(i + 1, startId + numPoints - 1);
                int d = Mathf.Min(i + 2, startId + numPoints - 1);

                Vector3 p1 = points[b];
                Vector3 p2 = points[c];
                Vector3 m1 = 0.5f * (points[c] - points[a]);
                Vector3 m2 = 0.5f * (points[d] - points[b]);

                // ensure last segment handled correctly
                int segments = (i < startId + numPoints - 2) ? smoothing : smoothing + 1;

                for (int s = 0; s < segments; ++s)
                {
                    Vector4 pos = HermiteInterpolate(p1, p2, m1, m2, s / (float)smoothing);
                    Vector3 dir = Vector3.Normalize(HermiteTangent(p1, p2, m1, m2, s / (float)smoothing));

                    //Vector3 cur = frame.GetAxis(2);
                    //Note intentional Vector4.w component drop
                    Vector3 cur = frame.GetColumn(2);
                    float angle = Mathf.Acos(Vector3.Dot(cur, dir));

                    // if parallel then don't need to do anything
                    if (Mathf.Abs(angle) > 0.001f)
                    {
                        //  Quaternion q = Quaternion.AngleAxis(angle, Vector3.Normalize(Vector3.Cross(cur, dir)));
                        //  frame = Matrix4x4.TRS(Vector3.zero, q, Vector3.one) * frame;
                        frame = RotationMatrix(angle, Vector3.Normalize(Vector3.Cross(cur, dir))) * frame;
                    }

                    for (int cc = 0; cc < resolution; ++cc)
                    {
                        float angle2 = 2.0f * Mathf.PI / (float)resolution;

                        // transform position and normal to world space
                        Vector4 vv = frame * new Vector4(Mathf.Cos(angle2 * cc), Mathf.Sin(angle2 * cc), 0.0f, 0.0f);
                        //verticesOut[vId] = tr.InverseTransformPoint(vv * radius + pos);
                        verticesOut[vId] = tr.InverseTransformPoint(vv * m_radii[i - startId] + pos);
                        normalsOut[vId] = tr.InverseTransformDirection(vv);
                        uvsOut[vId] = new Vector2(cc * uvStepH, (i * segments + s) * uvStepV);
                        colorsOut[vId] = colors[i];
                        //uvsOut[vId] = new Vector2(s * uvStepH, i * uvStepV);
                        //UVs[j * numVertexColumns + i] = new Vector2(i * uvStepH, j * uvStepV);
                        vId++;
                    }

                }

            }
        }

        public void DrawRope(Transform tr, NativeArray<Vector4> points, NativeArray<Color> colors, int startId, int numPoints, ref Vector3[] verticesOut, ref Vector3[] normalsOut, ref Color[] colorsOut, ref Vector2[] uvsOut, float radius, int resolution, int smoothing)
        {

            if (numPoints < 2)
                return;

            Vector3 u, v;
            Vector3 w = Vector3.Normalize(points[1] - points[0]);

            BasisFromVector(w, out u, out v);

            Matrix4x4 frame = new Matrix4x4();
            frame.SetColumn(0, new Vector4(u.x, u.y, u.z, 0.0f));
            frame.SetColumn(1, new Vector4(v.x, v.y, v.z, 0.0f));
            frame.SetColumn(2, new Vector4(w.x, w.y, w.z, 0.0f));
            frame.SetColumn(3, new Vector4(0.0f, 0.0f, 0.0f, 1.0f));

            float uvStepH = m_textureU / (float)resolution;
            float uvStepV = m_textureV / ((float)numPoints * smoothing);

            int vId = 0;
            for (int i = startId; i < startId + numPoints - 1; ++i)
            {
                Vector3 next = new Vector3();

                if (i > 0)
                {
                    if (i < startId + numPoints - 1)
                        next = Vector3.Normalize(points[i + 1] - points[i - 1]);
                    else
                        next = Vector3.Normalize(points[i] - points[i - 1]);
                }

                int a = Mathf.Max(i - 1, 0);
                int b = i;
                int c = Mathf.Min(i + 1, startId + numPoints - 1);
                int d = Mathf.Min(i + 2, startId + numPoints - 1);

                Vector3 p1 = points[b];
                Vector3 p2 = points[c];
                Vector3 m1 = 0.5f * (points[c] - points[a]);
                Vector3 m2 = 0.5f * (points[d] - points[b]);

                // ensure last segment handled correctly
                int segments = (i < startId + numPoints - 2) ? smoothing : smoothing + 1;

                for (int s = 0; s < segments; ++s)
                {
                    Vector4 pos = HermiteInterpolate(p1, p2, m1, m2, s / (float)smoothing);
                    Vector3 dir = Vector3.Normalize(HermiteTangent(p1, p2, m1, m2, s / (float)smoothing));

                    //Vector3 cur = frame.GetAxis(2);
                    //Note intentional Vector4.w component drop
                    Vector3 cur = frame.GetColumn(2);
                    float angle = Mathf.Acos(Vector3.Dot(cur, dir));

                    // if parallel then don't need to do anything
                    if (Mathf.Abs(angle) > 0.001f)
                    {
                        //  Quaternion q = Quaternion.AngleAxis(angle, Vector3.Normalize(Vector3.Cross(cur, dir)));
                        //  frame = Matrix4x4.TRS(Vector3.zero, q, Vector3.one) * frame;
                        frame = RotationMatrix(angle, Vector3.Normalize(Vector3.Cross(cur, dir))) * frame;
                    }

                    for (int cc = 0; cc < resolution; ++cc)
                    {
                        float angle2 = 2.0f * Mathf.PI / (float)resolution;

                        // transform position and normal to world space
                        Vector4 vv = frame * new Vector4(Mathf.Cos(angle2 * cc), Mathf.Sin(angle2 * cc), 0.0f, 0.0f);
                        //verticesOut[vId] = tr.InverseTransformPoint(vv * radius + pos);
                        verticesOut[vId] = tr.InverseTransformPoint(vv * m_radii[i - startId] + pos);
                        normalsOut[vId] = tr.InverseTransformDirection(vv);
                        uvsOut[vId] = new Vector2(cc * uvStepH, (i * segments + s) * uvStepV);
                        colorsOut[vId] = colors[i];
                        //uvsOut[vId] = new Vector2(s * uvStepH, i * uvStepV);
                        //UVs[j * numVertexColumns + i] = new Vector2(i * uvStepH, j * uvStepV);
                        vId++;
                    }

                }

            }
        }

        public void SetRadiusMult(float mult)
        {
            m_radiusMult = mult;
        }


        private void OnDestroy()
        {
            if(m_verticesNative.IsCreated)
            {
                m_verticesNative.Dispose();
                m_normalsNative.Dispose();
                m_colorsNative.Dispose();
                m_uvsNative.Dispose();
                m_trianglesNative.Dispose();
                m_radiiNative.Dispose();
            }
        }

        // generate a rotation matrix around an axis, from PBRT p74
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Matrix4x4 RotationMatrix(float angle, Vector3 axis)
        {
            Vector3 a = Vector3.Normalize(axis);
            float s = Mathf.Sin(angle);
            float c = Mathf.Cos(angle);

            Matrix4x4 m = new Matrix4x4();

            //m[0, 0] = a.x * a.x + (1.0f - a.x * a.x) * c;
            //m[0, 1] = a.x * a.y * (1.0f - c) + a.z * s;
            //m[0, 2] = a.x * a.z * (1.0f - c) - a.y * s;
            //m[0, 3] = 0.0f;

            //m[1, 0] = a.x * a.y * (1.0f - c) - a.z * s;
            //m[1, 1] = a.y * a.y + (1.0f - a.y * a.y) * c;
            //m[1, 2] = a.y * a.z * (1.0f - c) + a.x * s;
            //m[1, 3] = 0.0f;

            //m[2, 0] = a.x * a.z * (1.0f - c) + a.y * s;
            //m[2, 1] = a.y * a.z * (1.0f - c) - a.x * s;
            //m[2, 2] = a.z * a.z + (1.0f - a.z * a.z) * c;
            //m[2, 3] = 0.0f;

            //m[3, 0] = 0.0f;
            //m[3, 1] = 0.0f;
            //m[3, 2] = 0.0f;
            //m[3, 3] = 1.0f;


            m[0, 0] = a.x * a.x + (1.0f - a.x * a.x) * c;
            m[1, 0] = a.x * a.y * (1.0f - c) + a.z * s;
            m[2, 0] = a.x * a.z * (1.0f - c) - a.y * s;
            m[3, 0] = 0.0f;

            m[0, 1] = a.x * a.y * (1.0f - c) - a.z * s;
            m[1, 1] = a.y * a.y + (1.0f - a.y * a.y) * c;
            m[2, 1] = a.y * a.z * (1.0f - c) + a.x * s;
            m[3, 1] = 0.0f;

            m[0, 2] = a.x * a.z * (1.0f - c) + a.y * s;
            m[1, 2] = a.y * a.z * (1.0f - c) - a.x * s;
            m[2, 2] = a.z * a.z + (1.0f - a.z * a.z) * c;
            m[3, 2] = 0.0f;

            m[0, 3] = 0.0f;
            m[1, 3] = 0.0f;
            m[2, 3] = 0.0f;
            m[3, 3] = 1.0f;

            return m;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector3 HermiteInterpolate(Vector3 a, Vector3 b, Vector3 t1, Vector3 t2, float t)
        {
            // blending weights
            float w1 = 1.0f - 3 * t * t + 2 * t * t * t;
            float w2 = t * t * (3.0f - 2.0f * t);
            float w3 = t * t * t - 2 * t * t + t;
            float w4 = t * t * (t - 1.0f);

            // return weighted combination
            return a * w1 + b * w2 + t1 * w3 + t2 * w4;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float HermiteInterpolate(float a, float b, float t1, float t2, float t)
        {
            // blending weights
            float w1 = 1.0f - 3 * t * t + 2 * t * t * t;
            float w2 = t * t * (3.0f - 2.0f * t);
            float w3 = t * t * t - 2 * t * t + t;
            float w4 = t * t * (t - 1.0f);

            // return weighted combination
            return a * w1 + b * w2 + t1 * w3 + t2 * w4;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector3 HermiteTangent(Vector3 a, Vector3 b, Vector3 t1, Vector3 t2, float t)
        {
            // first derivative blend weights
            float w1 = 6.0f * t * t - 6 * t;
            float w2 = -6.0f * t * t + 6 * t;
            float w3 = 3 * t * t - 4 * t + 1;
            float w4 = 3 * t * t - 2 * t;

            // weighted combination
            return a * w1 + b * w2 + t1 * w3 + t2 * w4;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void BasisFromVector(Vector3 w, out Vector3 u, out Vector3 v)
        {
            if (Mathf.Abs(w.x) > Mathf.Abs(w.y))
            {
                float invLen = 1.0f / Mathf.Sqrt(w.x * w.x + w.z * w.z);
                u = new Vector3(-w.z * invLen, 0.0f, w.x * invLen);
            }
            else
            {
                float invLen = 1.0f / Mathf.Sqrt(w.y * w.y + w.z * w.z);
                u = new Vector3(0.0f, w.z * invLen, -w.y * invLen);
            }

            v = Vector3.Cross(w, u);


        }

    }
}