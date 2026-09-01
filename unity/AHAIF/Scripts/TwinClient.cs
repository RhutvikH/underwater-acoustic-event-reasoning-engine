using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;

namespace AHAIF
{
    [Serializable] public class TwinEnvironment { public int sea_state; public float ssp_m_s; public float temperature_c; public float turbulence; }
    [Serializable] public class TwinSource { public float[] xyz; public string @class; public bool present; public bool active; }
    [Serializable] public class TwinNode {
        public string node_id; public float[] xyz; public string profile;
        public float battery_frac; public int level; public float wake; public float trust;
        public string event_class; public string explanation; public float energy_j;
        public bool authenticated; public string[] woke_neighbors; public int confirmations;
        public float snr_db; public string reason;
    }
    [Serializable] public class TwinLink { public string src; public string dst; public string kind; }
    [Serializable] public class TwinKpis { public int ticks, true_events, detections, false_alarms, collab_wakes, explanations; public float joules; }
    [Serializable] public class TwinState {
        public float t; public string scenario;
        public TwinEnvironment environment; public TwinSource source;
        public TwinNode[] nodes; public TwinLink[] links; public TwinKpis kpis;
    }

    public class TwinClient : MonoBehaviour
    {
        public string url = "http://127.0.0.1:8765/api/state";
        public float pollSeconds = 0.4f;
        public TwinState Latest { get; private set; }

        void Start() { StartCoroutine(Poll()); }

        IEnumerator Poll()
        {
            while (true)
            {
                using (var req = UnityWebRequest.Get(url))
                {
                    yield return req.SendWebRequest();
                    if (req.result == UnityWebRequest.Result.Success)
                    {
                        try { Latest = JsonUtility.FromJson<TwinState>(req.downloadHandler.text); }
                        catch (Exception e) { Debug.LogWarning("AHAIF JSON: " + e.Message); }
                    }
                }
                yield return new WaitForSeconds(pollSeconds);
            }
        }
    }
}
