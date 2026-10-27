using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using OpenCvSharp;

namespace SanctumHelper;

/// <summary>Developer tools for the view-cone reader (they run beside a running copy):
///   --cone-record dir seconds   saves the game's minimap ~30 times a second, to tune the reader on
///   --cone-replay dir           runs the reader over saved frames and writes cone.csv (and an image per frame)</summary>
static class ConeRecord
{
    public static void Record(string dir, double seconds)
    {
        Directory.CreateDirectory(dir);
        var settings = Settings.Load();
        var game = new GameWindow();
        if (!game.Refresh(settings.WindowTitle)) { File.WriteAllText(Path.Combine(dir, "error.txt"), "game window not found"); return; }
        var c = game.Client;
        var rc = Regions.ToClient(settings.MinimapRegion, Anchor.TopLeft, c.Width, c.Height);
        var sw = Stopwatch.StartNew();
        int i = 0;
        while (sw.Elapsed.TotalSeconds < seconds)
        {
            long t0 = sw.ElapsedMilliseconds;
            using var mm = ScreenGrabber.Grab(new Rect(c.X + rc.X, c.Y + rc.Y, rc.Width, rc.Height));
            Cv2.ImWrite(Path.Combine(dir, $"f{i++:0000}_{t0:000000}.png"), mm);
            int left = 33 - (int)(sw.ElapsedMilliseconds - t0);
            if (left > 0) System.Threading.Thread.Sleep(left);
        }
    }

    public static void Replay(string dir)
    {
        var hd = new HeadingDetector(Path.Combine(AppContext.BaseDirectory, "Assets", "icons", "player_arrow.png"));
        var csv = new StringBuilder("frame,ms,arrow,cone,conf,egg,cone_centroid" + Environment.NewLine);
        var icons = Path.Combine(AppContext.BaseDirectory, "Assets", "icons");
        Directory.CreateDirectory(Path.Combine(dir, "vis"));
        foreach (var f in Directory.GetFiles(dir, "f*.png").OrderBy(x => x))
        {
            using var mm = Cv2.ImRead(f);
            double ui = 1080.0 / 1080.0 * mm.Height / 210.0;
            var arrow = hd.Detect(mm, ui);
            var cone = HeadingDetector.FromViewCone(mm, ui, out double conf);
            var old = HeadingDetector.FromViewConeCentroid(mm, ui, out _);
            var name = Path.GetFileNameWithoutExtension(f);
            double egg = HeadingDetector.EggScore(mm, ui, icons);
            csv.AppendLine($"{name},{name.Split('_')[1]},{arrow?.ToString("0") ?? ""},{cone?.ToString("0.0") ?? ""},{conf:0.00},{egg:0.00},{old?.ToString("0.0") ?? ""}");
            using var vis = new Mat();
            Cv2.Resize(mm, vis, new Size(240, 240));
            var p = new Point(120, 120);
            if (cone is double b) Cv2.Line(vis, p, new Point(120 + 110 * Math.Sin(b * Math.PI / 180), 120 - 110 * Math.Cos(b * Math.PI / 180)), new Scalar(255, 0, 255), 2);
            Cv2.ImWrite(Path.Combine(dir, "vis", name + ".png"), vis);
        }
        File.WriteAllText(Path.Combine(dir, "cone.csv"), csv.ToString());
    }
}
