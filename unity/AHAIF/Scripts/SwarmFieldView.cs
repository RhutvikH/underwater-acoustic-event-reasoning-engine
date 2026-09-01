using System.Collections.Generic;
using UnityEngine;

namespace AHAIF
{
    /// Visualises the Python scientific twin: cheap hydrophone swarm + one source.
    public class SwarmFieldView : MonoBehaviour
    {
        public TwinClient client;
        public float metresToUnits = 0.01f;
        public Material nodeMaterial;
        public Material sourceMaterial;
        public Material linkMaterial;

        readonly Dictionary<string, Transform> nodes = new Dictionary<string, Transform>();
        Transform source;
        readonly List<LineRenderer> links = new List<LineRenderer>();

        void Awake()
        {
            if (client == null) client = GetComponent<TwinClient>();
        }

        void Update()
        {
            var s = client != null ? client.Latest : null;
            if (s == null || s.nodes == null) return;
            if (source == null)
            {
                var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                go.name = "AcousticSource";
                go.transform.SetParent(transform);
                source = go.transform;
            }
            if (s.source != null && s.source.xyz != null && s.source.xyz.Length >= 3)
            {
                source.position = ToU(s.source.xyz);
                source.localScale = Vector3.one * (0.4f + 0.15f * Mathf.Sin(Time.time * 4f));
            }
            foreach (var n in s.nodes)
            {
                if (n.xyz == null || n.xyz.Length < 3) continue;
                if (!nodes.TryGetValue(n.node_id, out var tr))
                {
                    var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                    go.name = n.node_id;
                    go.transform.SetParent(transform);
                    tr = go.transform;
                    nodes[n.node_id] = tr;
                }
                tr.position = ToU(n.xyz);
                tr.localScale = Vector3.one * (0.18f + 0.35f * n.wake);
            }
        }

        Vector3 ToU(float[] xyz)
        {
            // X east, Y north → Unity XZ; depth Z → Unity -Y so the sea surface is y=0.
            return new Vector3(xyz[0] * metresToUnits, -xyz[2] * metresToUnits, xyz[1] * metresToUnits);
        }
    }
}
