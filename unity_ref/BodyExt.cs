using System.Collections;
using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections.NotBurstCompatible;
using UnityEngine;
using Unity.Mathematics;

namespace DefKit
{
    public unsafe class BodyExt : MonoBehaviour
    {
        public int count;
        public Color[] colors;

        public Vector4[] normals;
        public Vector3[] texCoords;
        public int[] groups;

        public float[] radius;
        public float[] stiffness;
        public byte[] collisionsIgnore;

        public NativeList<Color> colorsNative;
        public NativeList<float> stiffnessNative;
        public NativeList<float> radiusNative;
        public NativeList<Vector4> normalsNative;
        public NativeList<Vector4> deltasNative;
        public NativeList<int> deltasCountNative;
        public NativeList<Vector3> texCoordsNative;
        public NativeList<byte> collisionsIgnoreNative;
        public NativeList<CollisionInfo> collisionsRespMagNative;
        public NativeList<float> occlusionNative;

        public Color* colorsNativePtr;
        public Vector4* normalsNativePtr;
        public Vector3* texCoordsNativePtr;
        public byte* collisionsIgnoreNativePtr;
        public float* stiffnessNativePtr;
        public bool debugNormals;
        public bool debugColIgnore;
        public bool debugTexCoords;
        public bool debugRadius;
        public int debugGroups = -1;

        public void Awake()
        {
            InitRuntime();
        }

        private void Start()
        {

        }



        public virtual void InitRuntime()
        {
            InitNativeArrays();

            ResizeNativeArrays(this.count);

            colorsNative.CopyFromNBC(colors);
            normalsNative.CopyFromNBC(normals);
            texCoordsNative.CopyFromNBC(texCoords);
            collisionsIgnoreNative.CopyFromNBC(collisionsIgnore);
            stiffnessNative.CopyFromNBC(stiffness);

            //if (radius == null || radius.Length != this.count)
            //    radius = new float[count];

            radiusNative.CopyFromNBC(radius);
            occlusionNative.CopyFromNBC(stiffness);
            //collisionsRespMagNative
        }

        public virtual void InitManagedArrays(int count)
        {
            this.count = count;
            this.groups = new int[count];
            this.stiffness = new float[count];
            this.colors = new Color[count];
            this.normals = new Vector4[count];
            this.texCoords = new Vector3[count];
            this.collisionsIgnore = new byte[count];
            this.radius = new float[count];
        }

        [ContextMenu("InitManagedArrays")]
        private void InitManagedArrays()
        {
            InitManagedArrays(GetComponent<Body>().count);
        }

        public virtual void InitNativeArrays()
        {
            this.colorsNative = new NativeList<Color>(this.count, Allocator.Persistent);
            this.texCoordsNative = new NativeList<Vector3>(this.count, Allocator.Persistent);
            this.normalsNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.deltasNative = new NativeList<Vector4>(this.count, Allocator.Persistent);
            this.deltasCountNative = new NativeList<int>(this.count, Allocator.Persistent);
            this.collisionsIgnoreNative = new NativeList<byte>(this.count, Allocator.Persistent);
            this.stiffnessNative = new NativeList<float>(this.count, Allocator.Persistent);
            this.radiusNative = new NativeList<float>(this.count, Allocator.Persistent);

            this.collisionsRespMagNative = new NativeList<CollisionInfo>(this.count, Allocator.Persistent);
            this.occlusionNative = new NativeList<float>(this.count, Allocator.Persistent);
            
            UpdateNativePointers();
        }

        public void ResizeNativeArrays(int newLength)
        {
            this.colorsNative.ResizeUninitialized(newLength);
            this.normalsNative.ResizeUninitialized(newLength);
            this.texCoordsNative.ResizeUninitialized(newLength);
            this.collisionsIgnoreNative.ResizeUninitialized(newLength);
            this.stiffnessNative.ResizeUninitialized(newLength);
            this.radiusNative.ResizeUninitialized(newLength);
            this.deltasNative.ResizeUninitialized(newLength);
            this.deltasCountNative.ResizeUninitialized(newLength);

            this.collisionsRespMagNative.ResizeUninitialized(newLength);
            this.occlusionNative.ResizeUninitialized(newLength);

            UpdateNativePointers();
        }

        public void UpdateNativePointers()
        {
            this.colorsNativePtr = (Color*)this.colorsNative.GetUnsafePtr<Color>();
            this.texCoordsNativePtr = (Vector3*)this.texCoordsNative.GetUnsafePtr<Vector3>();
            this.normalsNativePtr = (Vector4*)this.normalsNative.GetUnsafePtr<Vector4>();
            this.collisionsIgnoreNativePtr = (byte*)this.collisionsIgnoreNative.GetUnsafePtr<byte>();
            this.stiffnessNativePtr = (float*)this.stiffnessNative.GetUnsafePtr<float>();
        }


        public virtual void DisposeArrays()
        {
            this.colorsNative.Dispose();
            this.normalsNative.Dispose();
            this.texCoordsNative.Dispose();
            this.collisionsIgnoreNative.Dispose();
            this.stiffnessNative.Dispose();
            this.deltasNative.Dispose();
            this.deltasCountNative.Dispose();
            this.radiusNative.Dispose();

            this.collisionsRespMagNative.Dispose();
            this.occlusionNative.Dispose();

            this.colorsNativePtr = null;
            this.texCoordsNativePtr = null;
            this.normalsNativePtr = null;
            this.collisionsIgnoreNativePtr = null;
            this.stiffnessNativePtr = null;
        }

        //   private void OnDrawGizmosSelected()
        private void OnDrawGizmos()
        {
            if (debugNormals)
            {
                Body body = GetComponent<Body>();
                if (Application.isPlaying)
                {
                    if (body.positionsNative.IsCreated && normalsNative.IsCreated)
                    {
                        for (int i = 0; i < body.count; i++)
                        {
                            DebugViz3D.DrawArrow(body.positionsNative[i], normalsNative[i] * 0.05f, Color.blue);

                        }
                    }
                }
                else
                {
                    for (int i = 0; i < body.count; i++)
                    {
                        Debug.DrawRay(body.positions[i], normals[i] * 0.1f, Color.blue);
                    }
                }
            }

            if (debugColIgnore)
            {
                Body body = GetComponent<Body>();
                if (Application.isPlaying)
                {
                    if (body.positionsNative.IsCreated && collisionsIgnoreNative.IsCreated)
                    {
                        for (int i = 0; i < body.count; i++)
                        {
                            if (collisionsIgnoreNative[i] == 0)
                                DebugViz3D.DrawSphere(body.positionsNative[i], 0.01f, Color.green);
                            else
                                DebugViz3D.DrawSphere(body.positionsNative[i], 0.01f, Color.red);
                        }
                    }
                }
            }

            if (debugTexCoords)
            {
                Body body = GetComponent<Body>();
                if (Application.isPlaying)
                {
                    if (body.positionsNative.IsCreated)
                    {

                    }
                }
                else
                {
                    Gizmos.color = Color.magenta;
                    for (int i = 0; i < body.count; i++)
                    {
                        if (texCoords[i].sqrMagnitude > 0)
                        {
                            Gizmos.DrawSphere(body.transform.TransformPoint(body.positions[i]), 0.005f);
                        }
                    }
                }
            }

            if (debugGroups > -1)
            {
                Body body = GetComponent<Body>();
                Gizmos.color = Color.cyan;
                for (int i = 0; i < body.count; i++)
                {
                    if (groups[i] == debugGroups)
                        Gizmos.DrawSphere(body.transform.TransformPoint(body.positions[i]), 0.005f);
                }

            }

            if (debugRadius)
            {
                if (Application.isPlaying)
                {
                    Body body = GetComponent<Body>();
                    Gizmos.color = Color.red;
                    for (int i = 0; i < body.count; i++)
                    {
                        //Gizmos.DrawSphere(body.transform.TransformPoint(body.positions[i]), 0.005f);
                        DebugViz3D.DrawSphere(body.positionsNative[i], radiusNative[i] * 2.0f, Color.red);
                    }
                }
            }
        }

        private void OnDestroy()
        {
            DisposeArrays();
        }
    }
}
