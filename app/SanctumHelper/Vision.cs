using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using OpenCvSharp;

namespace SanctumHelper;

/// <summary>Where a layout's map image sits on screen: screen = image * Scale + (Tx, Ty),
/// in client pixels of the game window.</summary>
public sealed record MapPose(Layout Layout, double Scale, double Tx, double Ty, int Inliers, double Rotation)
{
    public Point2d ToScreen(double ix, double iy) => new(ix * Scale + Tx, iy * Scale + Ty);
}

/// <summary>Map art prepared for matching: the layout image composited on the dark
/// background the game draws it on, as grayscale.</summary>
public sealed class MapArt : IDisposable
{
    public Mat Gray { get; }
    public Mat Alpha { get; }
    /// <summary>Colour art on the map's dark background (made on first use).</summary>
    public Mat Bgr => bgr ??= MakeBgr();
    Mat? bgr;
    readonly string path;
    readonly int padW, padH;

    public MapArt(string path, int width = 0, int height = 0)
    {
        this.path = path; padW = width; padH = height;
        using var raw = Cv2.ImRead(path, ImreadModes.Unchanged);
        // Upper-floor images can be cropped at the bottom/right (empty tiles not shipped): pad
        // them to the ground floor's size so all floors share coordinates.
        using var rgba = new Mat();
        Cv2.CopyMakeBorder(raw, rgba, 0, Math.Max(0, height - raw.Height), 0, Math.Max(0, width - raw.Width), BorderTypes.Constant, Scalar.All(0));
        var ch = rgba.Split();
        Alpha = ch[3].Clone();
        using var bgr = new Mat();
        Cv2.Merge([ch[0], ch[1], ch[2]], bgr);
        using var g = new Mat();
        Cv2.CvtColor(bgr, g, ColorConversionCodes.BGR2GRAY);
        Gray = new Mat();
        // a = alpha/255; gray = g * a + bg * (1 - a), bg ~ the map panel's near-black
        using var a = new Mat();
        Alpha.ConvertTo(a, MatType.CV_32F, 1 / 255.0);
        using var gf = new Mat();
        g.ConvertTo(gf, MatType.CV_32F);
        Cv2.Subtract(gf, new Scalar(20), gf);
        Cv2.Multiply(gf, a, gf);
        Cv2.Add(gf, new Scalar(20), gf);
        gf.ConvertTo(Gray, MatType.CV_8U);
        foreach (var c in ch) c.Dispose();
    }

    Mat MakeBgr()
    {
        using var raw = Cv2.ImRead(path, ImreadModes.Unchanged);
        using var rgba = new Mat();
        Cv2.CopyMakeBorder(raw, rgba, 0, Math.Max(0, padH - raw.Height), 0, Math.Max(0, padW - raw.Width), BorderTypes.Constant, Scalar.All(0));
        var ch = rgba.Split();
        using var a = new Mat();
        ch[3].ConvertTo(a, MatType.CV_32F, 1 / 255.0);
        var outCh = new Mat[3];
        for (int i = 0; i < 3; i++)
        {
            using var f = new Mat();
            ch[i].ConvertTo(f, MatType.CV_32F);
            Cv2.Subtract(f, new Scalar(20), f);
            Cv2.Multiply(f, a, f);
            Cv2.Add(f, new Scalar(20), f);
            outCh[i] = new Mat();
            f.ConvertTo(outCh[i], MatType.CV_8U);
        }
        var m = new Mat();
        Cv2.Merge(outCh, m);
        foreach (var c in ch) c.Dispose();
        foreach (var c in outCh) c.Dispose();
        return m;
    }

    public void Dispose() { Gray.Dispose(); Alpha.Dispose(); bgr?.Dispose(); }
}

public sealed class MapArtCache(LayoutDb db)
{
    readonly ConcurrentDictionary<(int, int), MapArt> art = new();
    public MapArt Get(Layout l, int floor = 0) => art.GetOrAdd((l.Number, floor),
        _ => new MapArt(db.FloorPath(l, floor), l.ImageSize[0], l.ImageSize[1]));

    /// <summary>Lets go of the map art of every layout but the ones kept (all of them when none is): a run's
    /// candidates used to stay loaded for good, ~400 MB after a run. Freed 30 s later, so a frame still being
    /// matched against one isn't cut off; a layout needed again is simply loaded again.</summary>
    public void Trim(Func<int, bool> keep) => Later.Dispose(Later.Take(art, k => keep(k.Item1)));
}

/// <summary>Freeing per-layout images. They're OpenCV memory the .NET garbage collector doesn't see, so a cache
/// that is only dropped (not disposed) can sit at hundreds of MB until the process ends.</summary>
static class Later
{
    /// <summary>Removes and returns the entries not kept.</summary>
    public static List<V> Take<K, V>(ConcurrentDictionary<K, V> d, Func<K, bool> keep) where K : notnull
    {
        var taken = new List<V>();
        foreach (var k in d.Keys)
            if (!keep(k) && d.TryRemove(k, out var v)) taken.Add(v);
        return taken;
    }

    /// <summary>Disposes 30 s later, so a frame still being matched against one isn't cut off.</summary>
    public static void Dispose<T>(List<T> items) where T : IDisposable
    {
        if (items.Count == 0) return;
        System.Threading.Tasks.Task.Delay(30_000).ContinueWith(_ => { foreach (var d in items) d.Dispose(); });
    }
}

/// <summary>ORB features of each layout's map art, at two sizes so both a zoomed-in
/// minimap and a zoomed-out full map find matches.</summary>
public sealed class ArtFeatureCache(MapArtCache arts)
{
    public sealed class Features { public required Mat Desc; public required Point2f[] FullPts; }

    readonly ConcurrentDictionary<int, Features> cache = new();
    static readonly double[] ArtScales = [0.5, 0.25];

    public Features Get(Layout l) => cache.GetOrAdd(l.Number, _ =>
    {
        var gray = arts.Get(l).Gray;
        var pts = new List<Point2f>();
        var descs = new List<Mat>();
        using var orb = ORB.Create(3000, 1.2f, 8, 15, 0, 2, ORBScoreType.Harris, 31, 12);
        foreach (var s in ArtScales)
        {
            using var small = new Mat();
            Cv2.Resize(gray, small, new Size(), s, s, InterpolationFlags.Area);
            using var eq = Vision.Clahe(small);
            var d = new Mat();
            orb.DetectAndCompute(eq, null, out var k, d);
            pts.AddRange(k.Select(p => new Point2f((float)(p.Pt.X / s), (float)(p.Pt.Y / s))));
            descs.Add(d);
        }
        var all = new Mat();
        Cv2.VConcat(descs.Where(d => !d.Empty()).ToArray(), all);
        foreach (var d in descs) d.Dispose();
        return new Features { Desc = all, FullPts = pts.ToArray() };
    });

    public void Prepare(IEnumerable<Layout> layouts)
    {
        foreach (var l in layouts) Get(l);
    }
}

public static class Vision
{
    public static Mat Clahe(Mat gray)
    {
        using var c = Cv2.CreateCLAHE(2.0, new Size(8, 8));
        var o = new Mat();
        c.Apply(gray, o);
        return o;
    }

    /// <summary>Similarity transform query image -> map art, from ORB matches + RANSAC:
    /// art = [a -b; b a] * query + t. Returns null when there is no consistent fit.</summary>
    public static (double a, double b, double tx, double ty, int inliers, int matches)? Fit(
        KeyPoint[] kps, Mat desc, double queryScale, ArtFeatureCache.Features art, double ratio, double reproj)
    {
        if (art.Desc.Empty() || desc.Empty()) return null;
        using var bf = new BFMatcher(NormTypes.Hamming);
        var knn = bf.KnnMatch(desc, art.Desc, 2);
        var src = new List<Point2f>();
        var dst = new List<Point2f>();
        foreach (var m in knn)
        {
            if (m.Length < 2 || m[0].Distance > ratio * m[1].Distance) continue;
            var p = kps[m[0].QueryIdx].Pt;
            src.Add(new Point2f((float)(p.X / queryScale), (float)(p.Y / queryScale)));
            dst.Add(art.FullPts[m[0].TrainIdx]);
        }
        if (src.Count < 6) return null;
        using var inl = new Mat();
        using var M = Cv2.EstimateAffinePartial2D(InputArray.Create(src), InputArray.Create(dst), inl,
            RobustEstimationAlgorithms.RANSAC, reproj, 3000, 0.995, 10);
        if (M == null || M.Empty()) return null;
        return (M.At<double>(0, 0), M.At<double>(1, 0), M.At<double>(0, 2), M.At<double>(1, 2), Cv2.CountNonZero(inl), src.Count);
    }
}

/// <summary>Finds which layout the open in-game map shows and where it is drawn, by
/// matching ORB features of the screen against each candidate layout's map art.</summary>
public sealed class MapMatcher(ArtFeatureCache features)
{
    public string LastInfo { get; private set; } = "";

    public MapPose? Match(Mat frameBgr, IReadOnlyList<Layout> candidates, MapPose? previous = null)
    {
        if (candidates.Count == 0) return null;
        double f = Math.Min(1.0, 1080.0 / frameBgr.Height);
        using var gray = new Mat();
        Cv2.CvtColor(frameBgr, gray, ColorConversionCodes.BGR2GRAY);
        using var small = new Mat();
        Cv2.Resize(gray, small, new Size(), f, f, InterpolationFlags.Area);
        using var eq = Vision.Clahe(small);
        using var orb = ORB.Create(6000, 1.2f, 8, 15, 0, 2, ORBScoreType.Harris, 31, 12);
        using var desc = new Mat();
        orb.DetectAndCompute(eq, null, out var kps, desc);
        if (desc.Empty() || kps.Length < 30) { LastInfo = "too few features on screen"; return null; }

        MapPose? best = null;
        int second = 0;
        var info = new List<string>();
        // Try the previously matched layout first; if it still matches well, skip the rest.
        var order = previous != null ? candidates.OrderByDescending(c => c.Number == previous.Layout.Number).ToList() : candidates.ToList();
        foreach (var l in order)
        {
            // Fit screen -> art, then invert to get art -> screen.
            var fit = Vision.Fit(kps, desc, f, features.Get(l), 0.78, 4.0);
            if (fit == null) { info.Add($"{l.Number}:-"); continue; }
            var (a, b, tx, ty, inliers, matches) = fit.Value;
            double k = Math.Sqrt(a * a + b * b), rot = Math.Atan2(b, a) * 180 / Math.PI;
            info.Add($"{l.Number}:{inliers}/{matches} s{1 / k:0.00} r{rot:0.0}");
            // The game never rotates its full map, and its zoom keeps the art between
            // roughly 0.15x and 6x on screen.
            if (Math.Abs(rot) > 3 || k < 1 / 6.0 || k > 1 / 0.15) continue;
            double scale = 1 / k;
            if (best == null || inliers > best.Inliers)
            {
                second = best?.Inliers ?? second;
                best = new MapPose(l, scale, -tx * scale, -ty * scale, inliers, rot);
            }
            else second = Math.Max(second, inliers);
            if (previous != null && l.Number == previous.Layout.Number && inliers >= 40) break;
        }
        LastInfo = string.Join("  ", info);
        if (best == null || best.Inliers < 14 || best.Inliers < second * 1.5) return null;
        return best;
    }
}

/// <summary>Finds the player from the minimap once the layout is known. The minimap shows the
/// layout's map art north-up around the player, but only rooms already explored: everything
/// else is the 3D world showing through. So only the explored floor (light, unsaturated
/// pixels) is compared, as a floor/wall shape, against the layout's floor shape.</summary>
public sealed class MinimapTracker(MapArtCache arts)
{
    // Full searches run at half resolution; tracking near a known spot at full resolution,
    // which keeps the fit precise (less jitter).
    const double SearchWork = 0.5, TrackWork = 1.0;
    // Measured 0.60 minimap px per map-art px at 1080p on live frames (layouts 53 and 58).
    static readonly double[] SearchScales = [0.56, 0.58, 0.6, 0.62, 0.64];
    readonly ConcurrentDictionary<(int, int, double, double), Mat> artFloor = new();
    /// <summary>Which floor the last fix was on: 0 ground, 1+ upstairs.</summary>
    public int Floor { get; private set; }
    int pendingFloor, floorVotes;
    double? scale;
    Point2d? last, jumpCandidate;
    DateTime lastOk, seededAt;
    public double LastScore { get; private set; }
    public string LastInfo { get; private set; } = "";
    /// <summary>Where the filter has the player now (map-art px), or null when lost.</summary>
    public Point2d? LastPosition => last;

    public void Reset() { scale = null; last = null; LastScore = 0; vel = default; Floor = 0; }

    /// <summary>Lets go of the cached floors of every layout but the ones kept (all of them when none is).</summary>
    public void Release(Func<int, bool>? keep = null) => Later.Dispose(Later.Take(artFloor, k => keep?.Invoke(k.Item1) ?? false));

    // Motion filter (constant velocity, alpha-beta): follows without the lag of plain smoothing,
    // carries through a missed fix, and refuses implausible jumps. Units: map-art px and px/s.
    Point2d vel;
    /// <summary>Current walking velocity (map-art px per second), for smooth drawing between fixes.</summary>
    public Point2d Velocity => vel;
    DateTime filterTime;

    /// <summary>Start tracking from a known spot (the arrow on the full map). A whole-map search
    /// is ambiguous when a layout repeats room shapes; a local search from a known spot isn't.</summary>
    public void Seed(Point2d pos) { last = pos; vel = default; seededAt = filterTime = DateTime.UtcNow; }

    /// <summary>Coloured parts of map icons (blue/cyan doors and eggs, orange door), dilated.</summary>
    public static Mat IconMask(Mat hsv, int margin)
    {
        var m = new Mat();
        Cv2.InRange(hsv, new Scalar(85, 56, 91), new Scalar(130, 255, 255), m);
        using var orange = new Mat();
        Cv2.InRange(hsv, new Scalar(0, 121, 91), new Scalar(22, 255, 255), orange);
        Cv2.BitwiseOr(m, orange, m);
        using var k = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(margin, margin));
        Cv2.Dilate(m, m, k);
        return m;
    }

    /// <summary>1 where the layout has walkable floor, 0 elsewhere, at a given minimap scale.</summary>
    Mat ArtFloor(Layout l, int floorIdx, double s, double work) => artFloor.GetOrAdd((l.Number, floorIdx, s, work), _ =>
    {
        var a = arts.Get(l, floorIdx);
        using var floor = new Mat();
        Cv2.Threshold(a.Gray, floor, 120, 1.0, ThresholdTypes.Binary);
        using var alpha = new Mat();
        Cv2.Threshold(a.Alpha, alpha, 128, 1.0, ThresholdTypes.Binary);
        Cv2.Multiply(floor, alpha, floor);
        using var f = new Mat();
        floor.ConvertTo(f, MatType.CV_32F);
        var small = new Mat();
        Cv2.Resize(f, small, new Size(), s * work, s * work, InterpolationFlags.Area);
        return small;
    });

    /// <summary>minimapBgr is the square crop centred on the player arrow; uiScale is the
    /// client height / 1080. Returns the player's position in map-art pixels.</summary>
    public Point2d? Locate(Mat minimapBgr, Layout layout, double uiScale)
    {
        if (minimapBgr.Width < 40) return null;
        bool tracking = last != null;
        double Work = tracking && scale != null ? TrackWork : SearchWork;
        using var small = new Mat();
        Cv2.Resize(minimapBgr, small, new Size(), Work / uiScale, Work / uiScale, InterpolationFlags.Area);
        using var hsv = new Mat();
        Cv2.CvtColor(small, hsv, ColorConversionCodes.BGR2HSV);
        using var disc = new Mat(small.Size(), MatType.CV_8U, Scalar.Black);
        var c = new Point(small.Width / 2, small.Height / 2);
        int r = (int)(Math.Min(small.Width, small.Height) * 0.44);
        Cv2.Circle(disc, c, r, Scalar.White, -1);
        Cv2.Circle(disc, c, Math.Max(4, r / 6), Scalar.Black, -1); // the player arrow
        // Explored floor on the minimap: grey, brighter than its walls. How bright depends on
        // the room's lighting (measured ~90 to ~200), so split walls/floor per frame (Otsu).
        using var grey = new Mat();
        Cv2.InRange(hsv, new Scalar(0, 0, 0), new Scalar(180, 40, 255), grey);
        Cv2.BitwiseAnd(grey, disc, grey);
        var hsvCh = hsv.Split();
        var greyIdx = new List<byte>();
        using (var idx = new Mat()) { Cv2.FindNonZero(grey, idx); for (int i = 0; i < idx.Rows; i++) { var pt = idx.At<Point>(i); greyIdx.Add(hsvCh[2].At<byte>(pt.Y, pt.X)); } }
        double cut = 131;
        if (greyIdx.Count > 50)
        {
            using var row = Mat.FromPixelData(1, greyIdx.Count, MatType.CV_8U, greyIdx.ToArray());
            using var tmp = new Mat();
            cut = Math.Max(70, Cv2.Threshold(row, tmp, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu));
        }
        using var floor8 = new Mat();
        Cv2.InRange(hsv, new Scalar(0, 0, cut + 1), new Scalar(180, 40, 255), floor8);
        // Some rooms have tan/beige floors: warm hue, moderate saturation (the orange door icon
        // and the yellow arrow are far more saturated). Warm-lit 3D world behind the minimap
        // has the same colours, but it is textured while the minimap's floor is flat, so tan
        // only counts where the brightness barely varies (local std < 10).
        using var tan = new Mat();
        Cv2.InRange(hsv, new Scalar(5, 0, Math.Max(cut, 110) + 1), new Scalar(30, 95, 255), tan);
        using (var v = new Mat())
        using (var mean = new Mat())
        using (var sq = new Mat())
        using (var flat = new Mat())
        {
            var hc = hsv.Split();
            hc[2].ConvertTo(v, MatType.CV_32F);
            foreach (var ch in hc) ch.Dispose();
            // Window sizes are in search-resolution px (Work 0.5) and scale with Work, so tracking
            // at full resolution judges the same physical area as the search did.
            int fw = Math.Max(3, (int)Math.Round(8 * Work));
            Cv2.Blur(v, mean, new Size(fw, fw));
            Cv2.Multiply(v, v, sq);
            Cv2.Blur(sq, sq, new Size(fw, fw));
            Cv2.Multiply(mean, mean, mean);
            Cv2.Subtract(sq, mean, sq);           // local variance
            Cv2.Threshold(sq, flat, 100, 255, ThresholdTypes.BinaryInv); // std < 10
            flat.ConvertTo(flat, MatType.CV_8U);
            Cv2.BitwiseAnd(tan, flat, tan);
        }
        Cv2.BitwiseOr(floor8, tan, floor8);
        Cv2.BitwiseAnd(floor8, disc, floor8);
        foreach (var ch in hsvCh) ch.Dispose();
        // Map icons (doors, eggs, nests...) sit on top of the floor and their white frames look
        // like floor: blank them out with a margin and leave them out of the comparison.
        // Measured on live frames: with icons in view, 0.45 -> 0.87 match.
        using var iconMask = IconMask(hsv, Math.Max(5, (int)(15 * Work)));
        floor8.SetTo(Scalar.Black, iconMask);
        Cv2.BitwiseNot(iconMask, iconMask);
        Cv2.BitwiseAnd(disc, iconMask, disc);
        int floorPx = Cv2.CountNonZero(floor8);
        if (floorPx < small.Width * small.Height / 40) { LastInfo = $"minimap: too little explored floor ({floorPx})"; return null; }
        // Compare the floor plus a band around it (its walls); ignore the unexplored rest.
        using var mask = new Mat();
        // The wall band is a fixed physical width: 5 px at search resolution, 9 at full. (A
        // fixed 5 px band made full-resolution tracking collapse in big open views: 0.87 -> 0.02.)
        int band = (int)Math.Round(10 * Work) | 1;
        using var k = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(band, band));
        Cv2.Dilate(floor8, mask, k);
        Cv2.BitwiseAnd(mask, disc, mask);
        using var tpl = new Mat();
        floor8.ConvertTo(tpl, MatType.CV_32F, 1 / 255.0);

        double[] scales = scale is double known && tracking ? [known] : SearchScales;
        (Point2d p, double score, double second, double s, int fl)? best = null;
        var perFloor = new Dictionary<int, (Point2d p, double score, double second, double s, int fl)>();
        // Upstairs the minimap shows the upper-floor drawing, so every floor is tried; the
        // current one first and with a small bonus so a tie doesn't flip floors.
        int floors = LayoutDb.FloorCount(layout);
        foreach (var fl in Enumerable.Range(0, floors).OrderBy(x => x == Floor ? 0 : 1))
        foreach (var s in scales)
        {
            var map = ArtFloor(layout, fl, s, Work);
            double kk = s * Work; // art px -> work px
            var roi = new Rect(0, 0, map.Width, map.Height);
            if (tracking)
            {
                // ~20 m around the last fix, growing while lost (players keep moving in menus/fades).
                double lost = Math.Min(8, (DateTime.UtcNow - (lastOk > seededAt ? lastOk : seededAt)).TotalSeconds);
                int half = (int)((120 + 60 * lost) * kk);
                roi = new Rect((int)(last!.Value.X * kk) - half - tpl.Width / 2, (int)(last.Value.Y * kk) - half - tpl.Height / 2,
                    2 * half + tpl.Width, 2 * half + tpl.Height).Intersect(roi);
            }
            if (roi.Width <= tpl.Width || roi.Height <= tpl.Height) continue;
            using var sub = new Mat(map, roi);
            using var res = new Mat();
            Cv2.MatchTemplate(sub, tpl, res, TemplateMatchModes.CCoeffNormed, mask);
            // Flat windows divide by ~0 and give NaN/±Inf: treat them as no match.
            Cv2.PatchNaNs(res, -1);
            Cv2.Threshold(res, res, 1.0, 0, ThresholdTypes.TozeroInv);
            Cv2.Max(res, new Scalar(-1), res);
            Cv2.MinMaxLoc(res, out _, out double max, out _, out Point loc);
            // Sub-pixel peak (parabola through the neighbours).
            double sx = 0, sy = 0;
            if (loc.X > 0 && loc.X < res.Width - 1)
            {
                float l0 = res.At<float>(loc.Y, loc.X - 1), c0 = res.At<float>(loc.Y, loc.X), r0 = res.At<float>(loc.Y, loc.X + 1);
                double d = l0 - 2 * c0 + r0; if (d < 0) sx = Math.Clamp(0.5 * (l0 - r0) / d, -0.5, 0.5);
            }
            if (loc.Y > 0 && loc.Y < res.Height - 1)
            {
                float u0 = res.At<float>(loc.Y - 1, loc.X), c0 = res.At<float>(loc.Y, loc.X), d0 = res.At<float>(loc.Y + 1, loc.X);
                double d = u0 - 2 * c0 + d0; if (d < 0) sy = Math.Clamp(0.5 * (u0 - d0) / d, -0.5, 0.5);
            }
            Cv2.Circle(res, loc, (int)(50 * kk), new Scalar(-1), -1);
            Cv2.MinMaxLoc(res, out _, out double max2, out _, out _);
            var cand = (new Point2d((roi.X + loc.X + sx + tpl.Width / 2.0) / kk, (roi.Y + loc.Y + sy + tpl.Height / 2.0) / kk), max, max2, s, fl);
            if (!perFloor.TryGetValue(fl, out var pf) || max > pf.score) perFloor[fl] = cand;
            double bonus = fl == Floor ? 0.03 : 0;
            if (best == null || max + bonus > best.Value.score + (best.Value.fl == Floor ? 0.03 : 0))
                best = cand;
        }
        if (best == null) return null;
        // Changing floors needs three fixes in a row that prefer the other floor; until then
        // stay on the current one (stairs and doorways briefly look like both).
        if (best.Value.fl != Floor && perFloor.ContainsKey(Floor))
        {
            floorVotes = pendingFloor == best.Value.fl ? floorVotes + 1 : 1;
            pendingFloor = best.Value.fl;
            if (floorVotes < 3) best = perFloor[Floor];
        }
        else floorVotes = 0;
        var b = best.Value;
        LastScore = b.score;
        LastInfo = $"minimap {b.score:0.00} (2nd {b.second:0.00}) at ({b.p.X:0},{b.p.Y:0}) s{b.s:0.00} level {b.fl} floor {floorPx}";
        // Right after a good fix a modest score is fine (the window is small); once lost, the
        // window is wide and look-alike spots appear, so ask for more.
        var now = DateTime.UtcNow;
        double dt = Math.Clamp((now - filterTime).TotalSeconds, 0.02, 0.5);
        bool recent = (now - lastOk).TotalSeconds < 1.0;
        var pred = last is Point2d lp0 ? new Point2d(lp0.X + vel.X * dt, lp0.Y + vel.Y * dt) : b.p;
        // In tight corridors little floor is visible and scores drop; a fix right where the
        // player is expected (within ~5 m) is still trustworthy at a lower score.
        bool nearPred = tracking && recent && Math.Abs(b.p.X - pred.X) + Math.Abs(b.p.Y - pred.Y) < 30;
        // Lost for a while (HUD covered, pillared rooms where little floor pattern shows): a fix
        // close to the last known spot that clearly beats every other spot is still the player.
        // Measured in layout 59's pillar room after a HUD cover: 0.44 / 0.42 (2nd 0.27-0.28), right spot.
        double lostFor = Math.Min(10, (now - (lastOk > seededAt ? lastOk : seededAt)).TotalSeconds);
        bool nearLast = tracking && !recent && Math.Abs(b.p.X - last!.Value.X) + Math.Abs(b.p.Y - last.Value.Y) < 120 + 40 * lostFor
            && b.second < b.score * 0.75;
        bool ok = tracking ? b.score >= (nearPred ? 0.35 : recent ? 0.45 : nearLast ? 0.38 : 0.55) : b.score >= 0.6 && b.second < b.score * 0.85;
        if (!ok)
        {
            filterTime = now;
            if (tracking && recent && (now - lastOk).TotalSeconds < 0.6)
            {
                // A missed fix mid-walk: keep moving along the current heading for a moment
                // instead of freezing, slowing down so a real stop doesn't overshoot far.
                vel = new Point2d(vel.X * 0.7, vel.Y * 0.7);
                last = pred;
                return pred;
            }
            // Lost (menu, loading screen, teleport). Keep the last spot for a while: menus and the
            // full map hide the minimap without moving the player (and the search window keeps
            // growing meanwhile). Give up after ~60 s: a whole-map search guesses badly in
            // layouts with repeated rooms.
            vel = default;
            if (tracking && (now - seededAt).TotalSeconds > 60 && (now - lastOk).TotalSeconds > 60) last = null;
            return null;
        }
        // Gate: a fix far from where we expect the player needs a second fix agreeing with it.
        double miss = Math.Abs(b.p.X - pred.X) + Math.Abs(b.p.Y - pred.Y);
        if (tracking && recent && miss > 60)
        {
            bool agrees = jumpCandidate is Point2d jc && Math.Abs(jc.X - b.p.X) + Math.Abs(jc.Y - b.p.Y) < 40;
            jumpCandidate = b.p;
            if (!agrees) { LastInfo += " (jump, waiting for confirmation)"; return pred; }
            vel = default; last = b.p; // confirmed (dash, teleport): snap
        }
        else if (!tracking || !recent)
        {
            vel = default; last = b.p; // (re)acquired
        }
        else
        {
            const double Alpha = 0.55, Beta = 0.12;
            double rx = b.p.X - pred.X, ry = b.p.Y - pred.Y;
            last = new Point2d(pred.X + Alpha * rx, pred.Y + Alpha * ry);
            vel = new Point2d(vel.X + Beta * rx / dt, vel.Y + Beta * ry / dt);
            double sp = Math.Sqrt(vel.X * vel.X + vel.Y * vel.Y);
            if (sp > 150) vel = new Point2d(vel.X * 150 / sp, vel.Y * 150 / sp); // ~25 m/s cap
        }
        jumpCandidate = null;
        lastOk = filterTime = now;
        scale = b.s;
        Floor = b.fl;
        return last;
    }
}

/// <summary>Aligns the full map from its two exit icons, which the game draws even under fog:
/// the orange door (where the team spawned) and the blue door. Their direction and distance
/// pick the layout, and the two points give the map's exact zoom and position. Revealed floor
/// must then land on the layout's art, which separates layouts with the same door direction.</summary>
public sealed class DoorLocator(MapArtCache arts, Mat? header, Mat? orangeIcon = null, Mat? blueIcon = null)
{
    /// <summary>Finds a door icon by matching the icon as the game draws it (cropped from 1080p
    /// screenshots). Colour blobs alone confuse the blue door with the light-blue egg and
    /// intrigue marks. Real doors score 0.84-1.0 (even half covered), anything else < 0.6.</summary>
    static Point2d? FindIcon(Mat frame1080, Mat? icon, Rect exclude, out double score)
    {
        score = 0;
        if (icon == null || icon.Empty() || frame1080.Width < icon.Width || frame1080.Height < icon.Height) return null;
        // Half resolution: 4x faster, still well above the icon's detail.
        const double H = 0.5;
        using var half = new Mat();
        Cv2.Resize(frame1080, half, new Size(), H, H, InterpolationFlags.Area);
        using var tpl = new Mat();
        Cv2.Resize(icon, tpl, new Size(), H, H, InterpolationFlags.Area);
        using var res = new Mat();
        Cv2.MatchTemplate(half, tpl, res, TemplateMatchModes.CCoeffNormed);
        // Ignore the title and zoom bar area.
        var ex = new Rect(0, 0, Math.Min(res.Width, (int)(exclude.Right * H)), Math.Min(res.Height, (int)(exclude.Bottom * H)));
        res[ex].SetTo(Scalar.All(-1));
        res[new Rect(0, 0, Math.Min(res.Width, (int)(150 * H)), res.Height)].SetTo(Scalar.All(-1));
        Cv2.MinMaxLoc(res, out _, out double max, out _, out Point loc);
        score = max;
        return max >= 0.75 ? new Point2d((loc.X + tpl.Width / 2.0) / H, (loc.Y + tpl.Height / 2.0) / H) : null;
    }

    public string LastInfo { get; private set; } = "";
    string iconScores = "";
    /// <summary>Door icons seen in the last frame (client pixels), even when no full fit was made.</summary>
    public Point2d? LastOrange { get; private set; }
    public Point2d? LastBlue { get; private set; }
    public bool LastMapOpen { get; private set; }

    /// <summary>The full map screen's title icon (top-left), so door-coloured things in normal
    /// gameplay are never mistaken for the map.</summary>
    public bool IsMapOpen(Mat small1080)
    {
        if (header == null || header.Empty()) return true;
        var area = new Rect(0, 0, Math.Min(small1080.Width, 400), Math.Min(small1080.Height, 200));
        using var roi = new Mat(small1080, area);
        using var res = new Mat();
        Cv2.MatchTemplate(roi, header, res, TemplateMatchModes.CCoeffNormed);
        Cv2.MinMaxLoc(res, out _, out double max, out _, out _);
        return max >= 0.8;
    }

    // HSV ranges measured on the game's map icons (OpenCV hue is 0-180).
    // Brightness floors are low because the map screen darkens towards its bottom edge
    // (measured: icons there lose ~40% brightness but keep their saturation).
    static readonly Scalar OrangeLo = new(5, 75, 90), OrangeHi = new(19, 255, 255);
    static readonly Scalar BlueLo = new(100, 100, 100), BlueHi = new(125, 255, 255);
    static readonly Scalar YellowLo = new(20, 70, 100), YellowHi = new(35, 255, 255);

    /// <summary>Map-art scale at the map's default (fully zoomed-out) level on a 1080p screen;
    /// measured 0.628-0.634 on layouts 53 and 58. Used to break ties between layouts whose
    /// doors point the same way (Normal 56/58/60 are all due east).</summary>
    const double DefaultScale = 0.632;

    /// <summary>Centre of the biggest blob of a colour, with its area, in frame pixels.</summary>
    static (Point2d c, int area)? Blob(Mat hsv, Scalar lo, Scalar hi, int minArea, int maxArea, Rect? exclude = null)
    {
        using var m = new Mat();
        Cv2.InRange(hsv, lo, hi, m);
        if (exclude is Rect ex) m[ex].SetTo(Scalar.Black);
        using var k = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(7, 7));
        Cv2.Dilate(m, m, k);
        using var labels = new Mat();
        using var stats = new Mat();
        using var cents = new Mat();
        int n = Cv2.ConnectedComponentsWithStats(m, labels, stats, cents);
        (Point2d, int)? best = null;
        for (int i = 1; i < n; i++)
        {
            int area = stats.At<int>(i, (int)ConnectedComponentsTypes.Area);
            if (area < minArea || area > maxArea) continue;
            if (best == null || area > best.Value.Item2)
                best = (new Point2d(cents.At<double>(i, 0), cents.At<double>(i, 1)), area);
        }
        return best;
    }

    /// <param name="preferred">A layout with independent evidence (the minimap tracker is
    /// matching it right now): wins a door tie instead of the default-zoom pick.</param>
    /// <param name="favoured">The detected difficulty's layouts: a tie between layouts of different
    /// difficulties goes to these (null when the difficulty reading is in doubt).</param>
    public MapPose? Match(Mat frameBgr, IReadOnlyList<Layout> candidates, out Point2d? arrow, Layout? preferred = null,
        IReadOnlyCollection<int>? favoured = null)
    {
        arrow = null;
        // Work at 1080p-equivalent resolution so the icon size limits hold on any screen.
        double f = 1080.0 / frameBgr.Height;
        using var small = new Mat();
        Cv2.Resize(frameBgr, small, new Size(), f, f, f < 1 ? InterpolationFlags.Area : InterpolationFlags.Linear);
        LastOrange = LastBlue = null;
        LastMapOpen = IsMapOpen(small);
        if (!LastMapOpen) { LastInfo = "map not open"; return null; }
        using var hsv = new Mat();
        Cv2.CvtColor(small, hsv, ColorConversionCodes.BGR2HSV);
        var hud = new Rect(0, 0, Math.Min(small.Width, 320), Math.Min(small.Height, 300)); // "Map" title / minimap corner
        (Point2d c, int area)? o = null, b = null;
        if (orangeIcon != null && blueIcon != null)
        {
            var oc = FindIcon(small, orangeIcon, new Rect(0, 0, 400, 120), out double os);
            var bc = FindIcon(small, blueIcon, new Rect(0, 0, 400, 120), out double bs);
            if (oc != null) o = (oc.Value, 1);
            if (bc != null) b = (bc.Value, 1);
            iconScores = $"icons o{os:0.00} b{bs:0.00}";
        }
        else
        {
            // Small minimum: the player arrow often covers part of the orange door at spawn.
            o = Blob(hsv, OrangeLo, OrangeHi, 30, 2500, hud);
            b = Blob(hsv, BlueLo, BlueHi, 30, 2500, hud);
        }
        // The game anchors map icons near their bottom edge: the door is ~11 px (1080p) below
        // the icon's colour centre (measured by floor matching on three map screenshots).
        const double IconAnchorY = 11;
        if (o != null) o = (new Point2d(o.Value.c.X, o.Value.c.Y + IconAnchorY), o.Value.area);
        if (b != null) b = (new Point2d(b.Value.c.X, b.Value.c.Y + IconAnchorY), b.Value.area);
        if (o != null) LastOrange = new Point2d(o.Value.c.X / f, o.Value.c.Y / f);
        if (b != null) LastBlue = new Point2d(b.Value.c.X / f, b.Value.c.Y / f);
        var y = Blob(hsv, YellowLo, YellowHi, 150, 4000, hud);
        if (y != null) arrow = new Point2d(y.Value.c.X / f, y.Value.c.Y / f);
        if (o == null || b == null) { LastInfo = $"doors: orange {(o == null ? "missing" : "ok")}, blue {(b == null ? "missing" : "ok")} ({iconScores})"; return null; }
        var O = o.Value.c; var B = b.Value.c;
        double wx = B.X - O.X, wy = B.Y - O.Y, dist = Math.Sqrt(wx * wx + wy * wy);
        if (dist < 25) { LastInfo = "doors overlap"; return null; }
        double ang = Math.Atan2(wx, -wy) * 180 / Math.PI;

        // Revealed floor: light, unsaturated pixels (fog is dark blue-grey).
        using var lit = new Mat();
        Cv2.InRange(hsv, new Scalar(0, 0, 141), new Scalar(180, 40, 255), lit);
        lit[hud].SetTo(Scalar.Black);
        int litCount = Cv2.CountNonZero(lit);

        var scored = new List<(Layout l, double angErr, double scale, double cover)>();
        foreach (var l in candidates)
        {
            var oi = l.Orange.Img; var bi = l.Blue.Img;
            double vx = bi[0] - oi[0], vy = bi[1] - oi[1], vlen = Math.Sqrt(vx * vx + vy * vy);
            double a = Math.Atan2(vx, -vy) * 180 / Math.PI;
            double err = Math.Abs(((a - ang) % 360 + 540) % 360 - 180);
            double s = dist / vlen;
            if (err > 8 || s < 0.1 || s > 6) continue;
            double cover = -1;
            // Too little revealed floor makes the coverage score noise (a 2k px glimpse once picked a
            // Nightmare layout); then the door angle and the default zoom decide.
            if (litCount > 5000)
            {
                using var M = new Mat(2, 3, MatType.CV_64F);
                M.Set(0, 0, s); M.Set(0, 1, 0.0); M.Set(0, 2, O.X - s * oi[0]);
                M.Set(1, 0, 0.0); M.Set(1, 1, s); M.Set(1, 2, O.Y - s * oi[1]);
                using var warped = new Mat();
                Cv2.WarpAffine(arts.Get(l).Alpha, warped, M, small.Size(), InterpolationFlags.Nearest);
                using var on = new Mat();
                Cv2.Threshold(warped, on, 128, 255, ThresholdTypes.Binary);
                using var both = new Mat();
                Cv2.BitwiseAnd(on, lit, both);
                cover = (double)Cv2.CountNonZero(both) / litCount;
            }
            scored.Add((l, err, s, cover));
        }
        LastInfo = $"doors {ang:0.0}° {dist / f:0}px lit {litCount}: " + string.Join("  ", scored.Select(x => $"{x.l.Number}:{x.angErr:0.0}° c{x.cover:0.00} s{x.scale:0.000}"));
        if (scored.Count == 0) return null;
        // With revealed floor, coverage decides; otherwise the door angle alone.
        var ranked = scored.OrderByDescending(x => x.cover >= 0 ? x.cover - x.angErr * 0.01 : -x.angErr).ToList();
        var best = ranked[0];
        if (best.cover >= 0 && best.cover < 0.6) return null;
        if (ranked.Count > 1)
        {
            var next = ranked[1];
            bool clear = best.cover >= 0 ? best.cover - next.cover > 0.15 : next.angErr - best.angErr > 2.5;
            if (!clear)
            {
                // Same direction, similar revealed floor: the one that fits the default zoom wins,
                // if the others are clearly off it (zooming in/out changes the scale by >20%).
                double Off(double sc) => Math.Abs(Math.Log(sc / DefaultScale));
                var close = ranked.Where(x => x.cover < 0 || x.cover >= best.cover - 0.15).OrderBy(x => Off(x.scale)).ToList();
                // Layouts sharing a door placement (32/56/58/60, 37/55) tie here; the minimap
                // already knows which one the player walks in, so that one is kept.
                int pi = preferred == null ? -1 : close.FindIndex(x => x.l.Number == preferred.Number);
                var fav = favoured == null ? [] : close.Where(x => favoured.Contains(x.l.Number)).ToList();
                if (pi >= 0)
                {
                    best = close[pi];
                    LastInfo += $"  (tie -> minimap's layout {best.l.Number})";
                }
                else if (fav.Count == 1)
                {
                    // Same doors in different difficulties (Chaos 32 / Normal 56, 58, 60): the
                    // party screen's difficulty decides.
                    best = fav[0];
                    LastInfo += $"  (tie -> difficulty's layout {best.l.Number})";
                }
                else if (Off(close[0].scale) < Math.Log(1.05) && (close.Count == 1 || Off(close[1].scale) > Math.Log(1.12)))
                {
                    best = close[0];
                    LastInfo += $"  (tie -> default zoom picks {best.l.Number})";
                }
                else { LastInfo += "  (ambiguous)"; return null; }
            }
        }
        double sc = best.scale / f;
        var bo = best.l.Orange.Img;
        return new MapPose(best.l, sc, O.X / f - sc * bo[0], O.Y / f - sc * bo[1], 100, 0);
    }
}

/// <summary>Exact placement of the full map: the explored floor visible on screen is matched
/// against the layout's floor shape, searching zoom and position around a first guess (doors,
/// last pose, pan). Works at any zoom and needs no doors on screen. Also removes the ~12 px
/// offset of guesses made from door icons, which the game anchors at their bottom edge.</summary>
public sealed class FloorRefiner(MapArtCache arts)
{
    const double K = 0.5; // analysis resolution relative to 1080p
    readonly ConcurrentDictionary<(int, int), Mat> artFloor = new();
    public string LastInfo { get; private set; } = "";
    public double LastScore { get; private set; }

    /// <summary>Lets go of the full-size floors of every layout but the ones kept (all of them when none is).</summary>
    public void Trim(Func<int, bool> keep) => Later.Dispose(Later.Take(artFloor, k => keep(k.Item1)));
    /// <summary>Floor the full map is showing: 0 ground, 1+ upstairs.</summary>
    public int Floor { get; private set; }

    Mat ArtFloor(Layout l, int level) => artFloor.GetOrAdd((l.Number, level), _ =>
    {
        var a = arts.Get(l, level);
        using var floor = new Mat();
        Cv2.Threshold(a.Gray, floor, 120, 1.0, ThresholdTypes.Binary);
        using var alpha = new Mat();
        Cv2.Threshold(a.Alpha, alpha, 128, 1.0, ThresholdTypes.Binary);
        Cv2.Multiply(floor, alpha, floor);
        var f = new Mat();
        floor.ConvertTo(f, MatType.CV_32F);
        return f;
    });

    /// <summary>Explored floor on the full map (light grey or tan), without the title, zoom bar
    /// and icons. frame1080 is the frame resized to 1080p height.</summary>
    static Mat ScreenFloor(Mat frame1080)
    {
        using var hsv = new Mat();
        Cv2.CvtColor(frame1080, hsv, ColorConversionCodes.BGR2HSV);
        var floor = new Mat();
        Cv2.InRange(hsv, new Scalar(0, 0, 141), new Scalar(180, 40, 255), floor);
        using var tan = new Mat();
        Cv2.InRange(hsv, new Scalar(5, 0, 141), new Scalar(30, 95, 255), tan);
        Cv2.BitwiseOr(floor, tan, floor);
        floor[new Rect(0, 0, Math.Min(floor.Width, 320), Math.Min(floor.Height, 300))].SetTo(Scalar.Black);
        floor[new Rect(0, 0, Math.Min(floor.Width, 140), floor.Height)].SetTo(Scalar.Black);
        using var icons = new Mat();
        Cv2.InRange(hsv, new Scalar(0, 101, 91), new Scalar(180, 255, 255), icons);
        using var k = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(25, 25));
        Cv2.Dilate(icons, icons, k);
        floor.SetTo(Scalar.Black, icons);
        using var icons2 = MinimapTracker.IconMask(hsv, 25);
        floor.SetTo(Scalar.Black, icons2);
        return floor;
    }

    /// <summary>Best pose near the guess, or null. spread = relative zoom range searched,
    /// window = position range searched in 1080p pixels.</summary>
    public MapPose? Refine(Mat frameBgr, MapPose guess, double spread, int window)
    {
        // Ground floor first; upstairs the game draws the upper-floor map instead.
        MapPose? bestPose = null; double bestScore = -1; string bestInfo = ""; int bestFloor = 0;
        int floors = LayoutDb.FloorCount(guess.Layout);
        foreach (var fl in Enumerable.Range(0, floors).OrderBy(x => x == Floor ? 0 : 1))
        {
            var p = RefineOn(frameBgr, guess, spread, window, fl);
            if (LastScore > bestScore) { bestScore = LastScore; bestPose = p; bestInfo = LastInfo; bestFloor = fl; }
            if (bestScore >= 0.6) break; // good enough on the current floor: skip the others
        }
        LastScore = Math.Max(0, bestScore);
        LastInfo = bestInfo + (floors > 1 ? $" level {bestFloor}" : "");
        if (bestPose != null) Floor = bestFloor;
        return bestPose;
    }

    MapPose? RefineOn(Mat frameBgr, MapPose guess, double spread, int window, int floorIdx)
    {
        double f = 1080.0 / frameBgr.Height;
        using var f1080 = new Mat();
        Cv2.Resize(frameBgr, f1080, new Size(), f, f, InterpolationFlags.Area);
        using var fl = ScreenFloor(f1080);
        int lit = Cv2.CountNonZero(fl);
        LastScore = 0;
        if (lit < 2000) { LastInfo = $"refine: little floor ({lit})"; return null; }
        using var flF = new Mat();
        fl.ConvertTo(flF, MatType.CV_32F, 1 / 255.0);
        using var small = new Mat();
        Cv2.Resize(flF, small, new Size(), K, K, InterpolationFlags.Area);
        using var dil = new Mat();
        using (var k = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(15, 15))) Cv2.Dilate(fl, dil, k);
        using var mask = new Mat();
        Cv2.Resize(dil, mask, small.Size(), 0, 0, InterpolationFlags.Nearest);
        var box = Cv2.BoundingRect(mask);
        if (box.Width < 20 || box.Height < 20) return null;
        using var tpl = new Mat(small, box);
        using var tm = new Mat(mask, box);

        // Guess in 1080p units.
        double s = guess.Scale * f, tx = guess.Tx * f, ty = guess.Ty * f; // (scale was divided: broke 4K)
        double x0 = box.X / K, y0 = box.Y / K; // template origin on screen (1080p px)
        var art = ArtFloor(guess.Layout, floorIdx);
        (double score, double sc, double tx, double ty)? best = null;
        void Search(double baseScale, double bx, double by, double spr, int steps, int win)
        {
            for (int i = 0; i < steps; i++)
            {
                double sc = baseScale * (steps == 1 ? 1 : Math.Exp(Math.Log(1 - spr) + (Math.Log(1 + spr) - Math.Log(1 - spr)) * i / (steps - 1)));
                using var A = new Mat();
                Cv2.Resize(art, A, new Size(), sc * K, sc * K, InterpolationFlags.Area);
                // Where the template's top-left should land in A under the base pose.
                double ex = (x0 - bx) / baseScale * sc * K, ey = (y0 - by) / baseScale * sc * K;
                int W = Math.Max(2, (int)(win * K));
                int P = Math.Max(tpl.Width, tpl.Height) + W;
                using var Ap = new Mat();
                Cv2.CopyMakeBorder(A, Ap, P, P, P, P, BorderTypes.Constant, Scalar.All(0));
                var roi = new Rect((int)ex + P - W, (int)ey + P - W, tpl.Width + 2 * W, tpl.Height + 2 * W)
                    .Intersect(new Rect(0, 0, Ap.Width, Ap.Height));
                if (roi.Width < tpl.Width || roi.Height < tpl.Height) continue;
                using var r = new Mat();
                Cv2.MatchTemplate(new Mat(Ap, roi), tpl, r, TemplateMatchModes.CCoeffNormed, tm);
                Cv2.PatchNaNs(r, -1);
                Cv2.Threshold(r, r, 1.0, 0, ThresholdTypes.TozeroInv);
                Cv2.Max(r, new Scalar(-1), r);
                Cv2.MinMaxLoc(r, out _, out double mx, out _, out Point loc);
                if (best == null || mx > best.Value.score)
                {
                    double ax = (roi.X + loc.X - P) / (sc * K), ay = (roi.Y + loc.Y - P) / (sc * K);
                    best = (mx, sc, x0 - sc * ax, y0 - sc * ay);
                }
            }
        }
        // Coarse around the guess, then fine around the best coarse result.
        Search(s, tx, ty, spread, spread > 0.1 ? 15 : 7, window);
        if (best == null) return null;
        var c = best.Value;
        Search(c.sc, c.tx, c.ty, 0.02, 5, 12);
        var b = best.Value;
        LastScore = b.score;
        LastInfo = $"refine {b.score:0.00} s{b.sc / f:0.000} (guess s{guess.Scale:0.000}) lit {lit}";
        if (b.score < 0.35) return null;
        return new MapPose(guess.Layout, b.sc / f, b.tx / f, b.ty / f, Math.Max(guess.Inliers, 60), 0);
    }
}

/// <summary>Finds the player on the island (outside the Sanctum) from the minimap. Unlike the
/// Sanctum there is no fog: the minimap is the island art drawn north-up at ~1.2x, so plain
/// grayscale matching works (measured 0.73 at the right entrance on a live frame).</summary>
public sealed class IslandTracker(MapArtCache arts)
{
    // Colour matching: the island art's colours (grass, sand, sea, paths) are what make places
    // tell apart; greyscale lost to the view cone in rainy green views (0.56 vs 0.60 in colour).
    const double SearchWork = 0.35, TrackWork = 0.7;
    // The island minimap is always drawn at 1.2x the map art: 504 of ~580 logged fixes were at
    // 1.20, and the 1.10 / 1.30 ones were look-alike places. Only that zoom is searched.
    static readonly double[] SearchScales = [1.2];
    readonly ConcurrentDictionary<(double, double), Mat> art = new();
    double? scale;
    double cropX, cropY;
    Point2d? lastFullBest;
    Point2d? last, jumpCandidate;
    Point2d vel;
    DateTime lastOk, filterTime;
    public double LastScore { get; private set; }
    public string LastInfo { get; private set; } = "";
    public bool Tracking => last != null && (DateTime.UtcNow - lastOk).TotalSeconds < 20;
    public Point2d Velocity => vel;

    public void Reset() { scale = null; last = null; vel = default; }

    /// <summary>Start from a known spot (the arrow on the full island map).</summary>
    public void Seed(Point2d pos) { last = pos; vel = default; lastOk = filterTime = DateTime.UtcNow; }

    Mat Art(Layout island, double s, double work) => art.GetOrAdd((s, work), _ =>
    {
        var m = new Mat();
        Cv2.Resize(arts.Get(island).Bgr, m, new Size(), s * work, s * work, InterpolationFlags.Area);
        return m;
    });

    /// <summary>Returns the player's position in island-art pixels, or null. A whole-island
    /// search is slow (~1 s), so the caller decides when one is allowed.</summary>
    public Point2d? Locate(Mat minimapBgr, Layout island, double uiScale, bool allowFullSearch)
    {
        bool tracking = Tracking;
        if (!tracking && !allowFullSearch) return null;
        double work = tracking && scale != null ? TrackWork : SearchWork;
        using var small = new Mat();
        Cv2.Resize(minimapBgr, small, new Size(), work / uiScale, work / uiScale, InterpolationFlags.Area);
        var gray = small; // colour template (the name is kept from the greyscale version)
        using var hsv = new Mat();
        Cv2.CvtColor(small, hsv, ColorConversionCodes.BGR2HSV);
        using var mask = new Mat(small.Size(), MatType.CV_8U, Scalar.Black);
        var c = new Point(small.Width / 2, small.Height / 2);
        int r = (int)(Math.Min(small.Width, small.Height) * 0.42);
        Cv2.Circle(mask, c, r, Scalar.White, -1);
        Cv2.Circle(mask, c, Math.Max(5, r / 4), Scalar.Black, -1); // the player arrow / egg marker
        // Not the Sanctum's icon filter: it blanks blue/cyan, which on the island is the sea,
        // the very thing that makes shores and docks recognisable.
        using (var icons = new Mat())
        using (var vivid = new Mat())
        {
            Cv2.InRange(hsv, new Scalar(0, 121, 91), new Scalar(180, 255, 255), vivid); // icons: vivid colours
            using var k5 = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(Math.Max(3, (int)(9 * work)), Math.Max(3, (int)(9 * work))));
            Cv2.Dilate(vivid, icons, k5);
            // The white view cone (and weather icon): near-white pixels. Measured on rainy and
            // clear frames: right spot 0.58 -> 0.68 vs runner-up ~0.50.
            using var white = new Mat();
            Cv2.InRange(hsv, new Scalar(0, 0, 201), new Scalar(180, 44, 255), white);
            using var k7 = Cv2.GetStructuringElement(MorphShapes.Rect, new Size(Math.Max(3, (int)(7 * work)), Math.Max(3, (int)(7 * work))));
            Cv2.Dilate(white, white, k7);
            Cv2.BitwiseOr(icons, white, icons);
            mask.SetTo(Scalar.Black, icons);
        }
        double[] scales = tracking && scale is double k0 ? [k0] : SearchScales;
        var shortlist = new List<(Point2d p, double s, double coarse)>();
        (Point2d p, double score, double second, double s)? best = null;
        foreach (var s in scales)
        {
            double kk = s * work;
            Mat map;
            Rect roi;
            Mat? owned = null;
            if (tracking)
            {
                // Only the neighbourhood, cut from the full-resolution art and scaled on the fly.
                var full = arts.Get(island).Bgr;
                double lost = Math.Min(8, (DateTime.UtcNow - lastOk).TotalSeconds);
                int halfArt = (int)(100 + 60 * lost + gray.Width / 2.0 / kk);
                var src = new Rect((int)last!.Value.X - halfArt, (int)last.Value.Y - halfArt, 2 * halfArt, 2 * halfArt).Intersect(new Rect(0, 0, full.Width, full.Height));
                owned = new Mat();
                Cv2.Resize(new Mat(full, src), owned, new Size(), kk, kk, InterpolationFlags.Area);
                map = owned;
                roi = new Rect(0, 0, map.Width, map.Height);
                // Positions below are in "scaled full map" units: shift by the crop origin.
                cropX = src.X * kk; cropY = src.Y * kk;
            }
            else { map = Art(island, s, work); roi = new Rect(0, 0, map.Width, map.Height); cropX = cropY = 0; }
            if (roi.Width <= gray.Width || roi.Height <= gray.Height) { owned?.Dispose(); continue; }
            using var res = new Mat();
            Cv2.MatchTemplate(new Mat(map, roi), gray, res, TemplateMatchModes.CCoeffNormed, mask);
            owned?.Dispose();
            Cv2.PatchNaNs(res, -1);
            Cv2.Threshold(res, res, 1.0, 0, ThresholdTypes.TozeroInv);
            Cv2.Max(res, new Scalar(-1), res);
            Cv2.MinMaxLoc(res, out _, out double max, out _, out Point loc);
            if (!tracking)
            {
                // Shortlist the best few spots for a sharper second look.
                using var rc = res.Clone();
                for (int k = 0; k < 5; k++)
                {
                    Cv2.MinMaxLoc(rc, out _, out double pv, out _, out Point pl);
                    if (pv < 0.35) break;
                    shortlist.Add((new Point2d((cropX + roi.X + pl.X + gray.Width / 2.0) / kk, (cropY + roi.Y + pl.Y + gray.Height / 2.0) / kk), s, pv));
                    Cv2.Circle(rc, pl, (int)(60 * kk), new Scalar(-1), -1);
                }
            }
            Cv2.Circle(res, loc, (int)(60 * kk), new Scalar(-1), -1);
            Cv2.MinMaxLoc(res, out _, out double max2, out _, out _);
            if (best == null || max > best.Value.score)
                best = (new Point2d((cropX + roi.X + loc.X + gray.Width / 2.0) / kk, (cropY + roi.Y + loc.Y + gray.Height / 2.0) / kk), max, max2, s);
        }
        if (!tracking && shortlist.Count > 0)
        {
            // Second look at double resolution around each shortlisted spot: separates places
            // the coarse search can't (rainy views: 0.54 vs 0.52 coarse).
            using var fineSmall = new Mat();
            Cv2.Resize(minimapBgr, fineSmall, new Size(), TrackWork / uiScale, TrackWork / uiScale, InterpolationFlags.Area);
            using var fineMask = new Mat();
            Cv2.Resize(mask, fineMask, fineSmall.Size(), 0, 0, InterpolationFlags.Nearest);
            var full = arts.Get(island).Bgr;
            var fine = new List<(Point2d p, double score, double s)>();
            foreach (var (p, s0, _) in shortlist.OrderByDescending(x => x.coarse).Take(6))
            {
                double kk = s0 * TrackWork;
                int halfArt = (int)(40 + fineSmall.Width / 2.0 / kk);
                var src = new Rect((int)p.X - halfArt, (int)p.Y - halfArt, 2 * halfArt, 2 * halfArt).Intersect(new Rect(0, 0, full.Width, full.Height));
                using var crop = new Mat();
                Cv2.Resize(new Mat(full, src), crop, new Size(), kk, kk, InterpolationFlags.Area);
                if (crop.Width <= fineSmall.Width || crop.Height <= fineSmall.Height) continue;
                using var r2 = new Mat();
                Cv2.MatchTemplate(crop, fineSmall, r2, TemplateMatchModes.CCoeffNormed, fineMask);
                Cv2.PatchNaNs(r2, -1);
                Cv2.Threshold(r2, r2, 1.0, 0, ThresholdTypes.TozeroInv);
                Cv2.MinMaxLoc(r2, out _, out double fv, out _, out Point fl);
                fine.Add((new Point2d(src.X + (fl.X + fineSmall.Width / 2.0) / kk, src.Y + (fl.Y + fineSmall.Height / 2.0) / kk), fv, s0));
            }
            var ranked = fine.OrderByDescending(x => x.score).ToList();
            if (ranked.Count > 0)
            {
                var top = ranked[0];
                // Runner-up: best shortlisted spot that is somewhere else.
                double second = ranked.Skip(1).Where(x => Math.Abs(x.p.X - top.p.X) + Math.Abs(x.p.Y - top.p.Y) > 60).Select(x => x.score).DefaultIfEmpty(0).Max();
                best = (top.p, top.score, second, top.s);
            }
        }
        if (best == null) return null;
        var b = best.Value;
        LastScore = b.score;
        LastInfo = $"island {b.score:0.00} (2nd {b.second:0.00}) at ({b.p.X:0},{b.p.Y:0}) s{b.s:0.00}";
        var now = DateTime.UtcNow;
        double dt = Math.Clamp((now - filterTime).TotalSeconds, 0.02, 0.5);
        // Only ever run when the minimap clearly shows the island, so the bar can be lower than
        // it had to be when Sanctum views were also searched.
        // A whole-island hit only starts a trial: the engine confirms it by tracking on locally
        // (a look-alike dock fails as soon as the player moves), so near-ties may pass here.
        // Open, even ground (a big grass field) matches weakly everywhere; a fix right where the
        // player is expected (~10 m) is still taken at a lower score, so tracking carries through.
        bool nearExpected = tracking && last is Point2d le && (now - lastOk).TotalSeconds < 1.5
            && Math.Abs(b.p.X - (le.X + vel.X * dt)) + Math.Abs(b.p.Y - (le.Y + vel.Y * dt)) < 40;
        bool ok = tracking ? b.score >= (nearExpected ? 0.30 : 0.42) : b.score >= 0.6 && b.second < b.score * 0.97;
        if (!tracking)
        {
            // Look-alike places (the island has several T-shaped docks) can tie a single search,
            // but the true spot wins search after search: two in a row at the same place is enough.
            // 0.5: open ground scores ~0.5 even at the right spot; the engine's local-tracking
            // trial still has to confirm a hit before the map moves there.
            bool repeat = lastFullBest is Point2d lf && Math.Abs(lf.X - b.p.X) + Math.Abs(lf.Y - b.p.Y) < 80 && b.score >= 0.5;
            lastFullBest = b.p;
            ok |= repeat;
        }
        if (!ok) { filterTime = now; return null; }
        var pred = last is Point2d lp && tracking ? new Point2d(lp.X + vel.X * dt, lp.Y + vel.Y * dt) : b.p;
        if (tracking && Math.Abs(b.p.X - pred.X) + Math.Abs(b.p.Y - pred.Y) > 80)
        {
            bool agrees = jumpCandidate is Point2d jc && Math.Abs(jc.X - b.p.X) + Math.Abs(jc.Y - b.p.Y) < 40;
            jumpCandidate = b.p;
            if (!agrees) return pred;
            last = b.p; vel = default;
        }
        else if (!tracking) { last = b.p; vel = default; }
        else
        {
            double rx = b.p.X - pred.X, ry = b.p.Y - pred.Y;
            last = new Point2d(pred.X + 0.55 * rx, pred.Y + 0.55 * ry);
            vel = new Point2d(vel.X + 0.12 * rx / dt, vel.Y + 0.12 * ry / dt);
            double sp = Math.Sqrt(vel.X * vel.X + vel.Y * vel.Y);
            if (sp > 120) vel = new Point2d(vel.X * 120 / sp, vel.Y * 120 / sp);
        }
        jumpCandidate = null;
        lastOk = filterTime = now;
        scale = b.s;
        return last;
    }
}

/// <summary>Which way the player faces, read from the minimap's arrow: the game's own arrow
/// sprite (points north) is rotated in steps and matched at the minimap centre. Measured on
/// live frames: best fit 0.81-0.91 at 0.40x sprite size at 1080p.</summary>
/// <summary>Finds the game's own map icons (portal doors, the Egg Boat, the egg marker) on the
/// minimap, using the sprites with their transparency as the mask.</summary>
public static class MapIcons
{
    static readonly System.Collections.Concurrent.ConcurrentDictionary<(string, int), (Mat bgr, Mat mask)> cache = new();

    static (Mat bgr, Mat mask) Sprite(string path, int size) => cache.GetOrAdd((path, size), _ =>
    {
        using var rgba = Cv2.ImRead(path, ImreadModes.Unchanged);
        using var small = new Mat();
        Cv2.Resize(rgba, small, new Size(size, size), 0, 0, InterpolationFlags.Area);
        var ch = small.Split();
        var bgr = new Mat();
        Cv2.Merge([ch[0], ch[1], ch[2]], bgr);
        var mask = new Mat();
        Cv2.Threshold(ch[3], mask, 128, 255, ThresholdTypes.Binary);
        foreach (var c in ch) c.Dispose();
        return (bgr, mask);
    });

    /// <summary>Matches of an icon in a BGR image already scaled to 1080p, strongest first, at
    /// least `apart` px from each other. sizes: icon widths to try (px).</summary>
    public static List<(Point2d at, double score, int size)> Find(Mat bgr, string iconPath, int[] sizes, double minScore, int max = 4, int apart = 14)
    {
        var all = new List<(Point2d at, double score, int size)>();
        foreach (var size in sizes)
        {
            var (tpl, mask) = Sprite(iconPath, size);
            if (tpl.Empty() || bgr.Width <= size || bgr.Height <= size) continue;
            using var res = new Mat();
            Cv2.MatchTemplate(bgr, tpl, res, TemplateMatchModes.CCoeffNormed, mask);
            Cv2.PatchNaNs(res, -1);
            Cv2.Threshold(res, res, 1.0, 0, ThresholdTypes.TozeroInv);
            for (int k = 0; k < max; k++)
            {
                Cv2.MinMaxLoc(res, out _, out double v, out _, out Point loc);
                if (v < minScore) break;
                all.Add((new Point2d(loc.X + size / 2.0, loc.Y + size / 2.0), v, size));
                Cv2.Circle(res, loc, apart, new Scalar(-1), -1);
            }
        }
        var kept = new List<(Point2d at, double score, int size)>();
        foreach (var c in all.OrderByDescending(x => x.score))
            if (kept.All(k => Math.Abs(k.at.X - c.at.X) + Math.Abs(k.at.Y - c.at.Y) > apart) && kept.Count < max) kept.Add(c);
        return kept;
    }
}

public sealed class HeadingDetector
{
    static List<(Mat tpl, Mat mask)>? eggTemplates;

    /// <summary>How much the minimap's centre looks like the egg marker the game shows instead of
    /// the arrow while the player is twined with an egg (0-1). Two crops of the real marker (one
    /// over Sanctum floor, one over the island), the better one counts; the player-number badge on
    /// its lower right changes with the player, so it is left out of the comparison.</summary>
    public static double EggScore(Mat minimapBgr, double uiScale, string iconsDir)
    {
        eggTemplates ??= new[] { "hud_egg_marker.png", "hud_egg_marker2.png" }.Select(n =>
        {
            var t = Cv2.ImRead(System.IO.Path.Combine(iconsDir, n));
            var m = new Mat(t.Size(), MatType.CV_8U, Scalar.Black);
            if (!t.Empty())
            {
                Cv2.Circle(m, new Point(t.Width / 2, t.Height / 2), t.Width / 2 - 2, Scalar.White, -1);
                Cv2.Circle(m, new Point(t.Width - 3, t.Height - 3), t.Width / 3, Scalar.Black, -1); // number badge
            }
            return (t, m);
        }).Where(x => !x.t.Empty()).ToList();
        if (eggTemplates.Count == 0) return 0;
        // Only right at the centre, where the player's own marker sits (an egg lying nearby shows
        // the same marker a little off-centre), at 1080p scale (the template's).
        int half = (int)((15 + 12) * uiScale);
        var c = new Point(minimapBgr.Width / 2, minimapBgr.Height / 2);
        var roi = new Rect(c.X - half, c.Y - half, 2 * half, 2 * half).Intersect(new Rect(0, 0, minimapBgr.Width, minimapBgr.Height));
        using var crop0 = new Mat(minimapBgr, roi);
        using var crop = new Mat();
        Cv2.Resize(crop0, crop, new Size(), 1 / uiScale, 1 / uiScale, InterpolationFlags.Area);
        double best = 0;
        foreach (var (tpl, mask) in eggTemplates)
        {
            if (crop.Width <= tpl.Width || crop.Height <= tpl.Height) continue;
            using var res = new Mat();
            Cv2.MatchTemplate(crop, tpl, res, TemplateMatchModes.CCoeffNormed, mask);
            Cv2.PatchNaNs(res, -1);
            Cv2.Threshold(res, res, 1.0, 0, ThresholdTypes.TozeroInv);
            Cv2.MinMaxLoc(res, out _, out double max, out _, out _);
            best = Math.Max(best, max);
        }
        return best;
    }

    static readonly System.Collections.Concurrent.ConcurrentDictionary<(int, int), int[]> coneLuts = new();

    static int[] ConeLut(int w, int h, int bins) => coneLuts.GetOrAdd((w, h), _ =>
    {
        var lut = new int[w * h];
        double cx = w / 2.0, cy = h / 2.0, R = Math.Min(cx, cy) * 0.84;
        double r0 = R * 0.14, r1 = R * 0.55; // outside the arrow / egg marker, where the cone is brightest
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
            {
                double dx = x + 0.5 - cx, dy = y + 0.5 - cy, r = Math.Sqrt(dx * dx + dy * dy);
                double ang = (Math.Atan2(dx, -dy) * 180 / Math.PI + 360) % 360;
                lut[y * w + x] = r < r0 || r > r1 ? -1 : (int)(ang / (360.0 / bins)) % bins;
            }
        return lut;
    });

    /// <summary>Which way the camera looks, from the white view cone the minimap draws from the
    /// player: a soft near-white wedge (~70-90 deg), brightest close to the player. Works when
    /// the arrow can't be read (twined with an egg: the egg marker replaces it). Degrees
    /// clockwise from north, or null; confidence is how much the cone stands out (0-1).</summary>
    public static double? FromViewCone(Mat minimapBgr, double uiScale, out double confidence)
    {
        // The cone's two edges, read at full 1080p resolution in 1-degree steps: the middle between them is where the
        // camera looks. Bright floor or walls under the cone shift its brightest part (what the centroid reader below
        // follows, so the cone drifted) but not where it starts and ends. Falls back to the centroid reader when the
        // edges don't look like a cone.
        confidence = 0;
        using var small = new Mat();
        double k = 210.0 / Math.Min(minimapBgr.Width, minimapBgr.Height);
        Cv2.Resize(minimapBgr, small, new Size(), k, k, InterpolationFlags.Area);
        using var hsv = new Mat();
        Cv2.CvtColor(small, hsv, ColorConversionCodes.BGR2HSV);
        const int Bins = 360;
        var sum = new double[Bins];
        var n = new int[Bins];
        var lut = ConeLut(small.Width, small.Height, Bins);
        hsv.GetArray(out Vec3b[] px);
        for (int i = 0; i < px.Length; i++)
        {
            int b = lut[i];
            if (b < 0) continue;
            var v = px[i];
            if (v.Item1 < 50 && v.Item2 > 205) sum[b] += Math.Min(1, (v.Item2 - 205) / 35.0);
            n[b]++;
        }
        var raw = new double[Bins];
        for (int i = 0; i < Bins; i++) raw[i] = n[i] > 0 ? sum[i] / n[i] : 0;
        var prof = new double[Bins];                 // lightly smoothed (5 deg), so single pixels don't make edges
        for (int i = 0; i < Bins; i++)
        {
            double t = 0;
            for (int j = -2; j <= 2; j++) t += raw[((i + j) % Bins + Bins) % Bins];
            prof[i] = t / 5;
        }
        // where the cone is: the best 72-degree arc (as the centroid reader), and how much it stands out
        const int Win = 72;
        int best = 0; double bestSum = -1;
        for (int i = 0; i < Bins; i++)
        {
            double t = 0;
            for (int j = 0; j < Win; j++) t += prof[(i + j) % Bins];
            if (t > bestSum) { bestSum = t; best = i; }
        }
        double inside = bestSum / Win;
        var rest = Enumerable.Range(0, Bins).Where(i => ((i - best) % Bins + Bins) % Bins >= Win).Select(i => prof[i]).OrderBy(v => v).ToList();
        double outside = rest[rest.Count / 2];
        confidence = Math.Clamp(inside - outside, 0, 1);
        if (inside < 0.25 || inside < outside * 2 + 0.1) return null;
        // the edges: from the arc's middle, outwards until the profile falls below half-way between the cone's
        // plateau and the floor around it (with sub-degree interpolation)
        var arc = Enumerable.Range(0, Win).Select(j => prof[(best + j) % Bins]).OrderBy(v => v).ToList();
        double plateau = arc[(int)(arc.Count * 0.75)];
        double level = outside + 0.5 * (plateau - outside);
        int mid = (best + Win / 2) % Bins;
        double? Edge(int step)
        {
            for (int d = 0; d < 100; d++)
            {
                int a = ((mid + step * d) % Bins + Bins) % Bins, b = ((mid + step * (d + 1)) % Bins + Bins) % Bins;
                if (prof[a] >= level && prof[b] < level)
                    return mid + step * (d + (prof[a] - level) / Math.Max(1e-9, prof[a] - prof[b]));
            }
            return null;
        }
        if (Edge(-1) is double left && Edge(1) is double right && right - left is >= 35 and <= 140)
            return (((left + right) / 2 + 0.5) % 360 + 360) % 360;      // + 0.5: bin i covers i..i+1 degrees
        return FromViewConeCentroid(minimapBgr, uiScale, out confidence);
    }

    /// <summary>The previous reader: the brightness-weighted centre of the brightest 72-degree arc, at half
    /// resolution. Kept as the fallback, and for comparing the two (--cone-replay writes both).</summary>
    public static double? FromViewConeCentroid(Mat minimapBgr, double uiScale, out double confidence)
    {
        confidence = 0;
        using var small = new Mat();
        double k = 105.0 / Math.Min(minimapBgr.Width, minimapBgr.Height); // half resolution of 1080p
        Cv2.Resize(minimapBgr, small, new Size(), k, k, InterpolationFlags.Area);
        using var hsv = new Mat();
        Cv2.CvtColor(small, hsv, ColorConversionCodes.BGR2HSV);
        const int Bins = 90; // 4 deg
        var sum = new double[Bins];
        var n = new int[Bins];
        // Which 4-degree slice each pixel of the ring near the player falls in (-1: outside the
        // ring), worked out once per size: the cone is read ~30 times a second.
        var lut = ConeLut(small.Width, small.Height, Bins);
        hsv.GetArray(out Vec3b[] px);
        for (int i = 0; i < px.Length; i++)
        {
            int b = lut[i];
            if (b < 0) continue;
            // Near-white: the cone lifts floor (~175-200) to ~230+ and leaves it unsaturated.
            var v = px[i];
            if (v.Item1 < 50 && v.Item2 > 205) sum[b] += Math.Min(1, (v.Item2 - 205) / 35.0);
            n[b]++;
        }
        var prof = new double[Bins];
        for (int i = 0; i < Bins; i++) prof[i] = n[i] > 0 ? sum[i] / n[i] : 0;
        // The best ~72 deg arc, then its brightness-weighted centre.
        const int Win = 18;
        int best = 0; double bestSum = -1;
        for (int i = 0; i < Bins; i++)
        {
            double t = 0;
            for (int j = 0; j < Win; j++) t += prof[(i + j) % Bins];
            if (t > bestSum) { bestSum = t; best = i; }
        }
        double inside = bestSum / Win;
        var rest = Enumerable.Range(0, Bins).Where(i => ((i - best) % Bins + Bins) % Bins >= Win).Select(i => prof[i]).OrderBy(v => v).ToList();
        double outside = rest[rest.Count / 2];
        confidence = Math.Clamp(inside - outside, 0, 1);
        if (inside < 0.25 || inside < outside * 2 + 0.1) return null;
        double sx = 0, sy = 0;
        for (int j = 0; j < Win; j++)
        {
            int i = (best + j) % Bins;
            double a = (i * 4 + 2) * Math.PI / 180;
            sx += prof[i] * Math.Sin(a); sy += prof[i] * Math.Cos(a);
        }
        return (Math.Atan2(sx, sy) * 180 / Math.PI + 360) % 360;
    }

    readonly List<(double angle, Mat bgr, Mat mask)> bank = new();
    public double LastScore { get; private set; }

    public HeadingDetector(string spritePath)
    {
        using var rgba = Cv2.ImRead(spritePath, ImreadModes.Unchanged);
        if (rgba.Empty() || rgba.Channels() != 4) return;
        using var small = new Mat();
        Cv2.Resize(rgba, small, new Size(), 0.40, 0.40, InterpolationFlags.Area);
        for (double a = 0; a < 360; a += 2.5)
        {
            using var M = Cv2.GetRotationMatrix2D(new Point2f(small.Width / 2f, small.Height / 2f), -a, 1);
            using var r = new Mat();
            Cv2.WarpAffine(small, r, M, small.Size());
            var ch = r.Split();
            var bgr = new Mat();
            Cv2.Merge([ch[0], ch[1], ch[2]], bgr);
            var mask = new Mat();
            Cv2.Threshold(ch[3], mask, 150, 255, ThresholdTypes.Binary);
            foreach (var c in ch) c.Dispose();
            bank.Add((a, bgr, mask));
        }
    }

    /// <summary>Degrees clockwise from north, or null. minimapBgr is the square minimap crop
    /// (centred on the arrow) at any resolution; uiScale = client height / 1080.</summary>
    /// <param name="hint">The heading a moment ago: then only ±22.5° around it is searched (a
    /// third of the matches), with the full circle as the fallback.</param>
    public double? Detect(Mat minimapBgr, double uiScale, double? hint = null)
    {
        if (bank.Count == 0) return null;
        // The arrow sits in the middle: a 48 px (1080p) square around the centre is enough.
        int half = (int)(24 * uiScale);
        var c = new Point(minimapBgr.Width / 2, minimapBgr.Height / 2);
        var roi = new Rect(c.X - half, c.Y - half, half * 2, half * 2).Intersect(new Rect(0, 0, minimapBgr.Width, minimapBgr.Height));
        using var crop0 = new Mat(minimapBgr, roi);
        using var crop = new Mat();
        Cv2.Resize(crop0, crop, new Size(48, 48), 0, 0, InterpolationFlags.Area);
        double Score(int i)
        {
            var (_, bgr, mask) = bank[i];
            if (crop.Width < bgr.Width || crop.Height < bgr.Height) return -1;
            using var res = new Mat();
            Cv2.MatchTemplate(crop, bgr, res, TemplateMatchModes.CCoeffNormed, mask);
            Cv2.PatchNaNs(res, -1);
            Cv2.Threshold(res, res, 1.0, 0, ThresholdTypes.TozeroInv);
            Cv2.MinMaxLoc(res, out _, out double max, out _, out _);
            return max;
        }
        int n = bank.Count, step = 6, best = 0;
        double bestScore = -1;
        if (hint is double h0)
        {
            // Near the last heading: every 7.5 deg within ±22.5, then fine around the best.
            int c0 = (int)Math.Round(h0 / 2.5) % n;
            for (int d = -9; d <= 9; d += 3) { int i = ((c0 + d) % n + n) % n; double sc = Score(i); if (sc > bestScore) { bestScore = sc; best = i; } }
            int mid = best;
            foreach (int d in new[] { -2, -1, 1, 2 }) { int i = ((mid + d) % n + n) % n; double sc = Score(i); if (sc > bestScore) { bestScore = sc; best = i; } }
            if (bestScore >= 0.65) { LastScore = bestScore; return bank[best].angle; }
            bestScore = -1; best = 0;
        }
        // Coarse (every 15 deg), then fine around the best (2.5 deg steps).
        for (int i = 0; i < n; i += step) { double sc = Score(i); if (sc > bestScore) { bestScore = sc; best = i; } }
        int coarse = best;
        for (int d = -step + 1; d < step; d++)
        {
            if (d == 0) continue;
            int i = ((coarse + d) % n + n) % n;
            double sc = Score(i);
            if (sc > bestScore) { bestScore = sc; best = i; }
        }
        LastScore = bestScore;
        return bestScore >= 0.65 ? bank[best].angle : null;
    }
}
