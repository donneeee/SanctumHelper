using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using OpenCvSharp;

namespace SanctumHelper;

/// <summary>What the overlay should draw right now.</summary>
/// <summary>The latest facing and camera directions, for the overlay to animate between frames

/// <summary>One-euro filter for an angle in degrees (Casiez et al.): smooths a lot while the value holds still and
/// less the faster it moves, so a resting cone stops jittering and a turning one doesn't trail behind.</summary>
public sealed class AngleFilter
{
    // 1.2 Hz when still; + 0.02 Hz per deg/s of turning (a 180 deg/s spin: ~4.8 Hz, about 30 ms of lag)
    const double MinCutoff = 1.2, Beta = 0.02, SpeedCutoff = 1.0;
    double value, speed;
    DateTime last;

    public double Reset(double angle, DateTime now) { value = angle; speed = 0; last = now; return value; }

    public double Next(double angle, DateTime now)
    {
        double dt = Math.Clamp((now - last).TotalSeconds, 1e-3, 0.5);
        last = now;
        double d = ((angle - value) % 360 + 540) % 360 - 180;     // the short way round
        speed += Alpha(SpeedCutoff, dt) * (d / dt - speed);
        value = ((value + Alpha(MinCutoff + Beta * Math.Abs(speed), dt) * d) % 360 + 360) % 360;
        return value;
    }

    static double Alpha(double cutoff, double dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
}

/// (written by the engine's heading loop, read on the UI thread).</summary>
public static class HeadingFeed
{
    static readonly object Gate = new();
    static double? heading, camera;
    static bool egg;
    static DateTime stamp;

    public static void Publish(double? h, double? c, bool eggForm)
    {
        lock (Gate) { heading = h; camera = c; egg = eggForm; stamp = DateTime.UtcNow; }
    }

    public static (double? heading, double? camera, DateTime stamp) Get()
    {
        lock (Gate) return (heading, camera, stamp);
    }

    /// <summary>The player is twined with an egg (the minimap shows the egg marker, not the arrow).</summary>
    public static bool Egg
    {
        get { lock (Gate) return egg && (DateTime.UtcNow - stamp).TotalSeconds < 1.5; }
    }
}

public sealed record OverlayFrame(
    Rect Client, bool Visible, Layout? Layout, MapPose? Pose, Point2d? Player, bool PlayerFresh,
    string? Difficulty, string Status, bool Mini = false, int Floor = 0, double? Heading = null,
    bool Big = false, Point2d? Velocity = null, DateTime Stamp = default, double? Camera = null);

/// <summary>Background loop: finds the game, reads the party screen, tracks the minimap,
/// and aligns the layout map while the hold key is down.</summary>
public sealed class Engine : IDisposable
{
    readonly LayoutDb db;
    readonly Settings settings;
    readonly KeyHook hook;
    // The party screen is in the game's language: read it with that language's OCR if Windows has it.
    OcrService ocr = new(Lang.OcrTag(Lang.GameCode));
    DifficultyDetector difficulty;

    /// <summary>The game language setting changed: read the party screen in the new one.</summary>
    public void GameLanguageChanged()
    {
        var o = new OcrService(Lang.OcrTag(Lang.GameCode));
        difficulty = new DifficultyDetector(o);
        ocr = o;
        Log.Write($"game language {Lang.GameCode}, OCR {o.Language}{(o.MissingLanguage != null ? $" ({o.MissingLanguage} not installed)" : "")}");
    }
    readonly MapArtCache arts;
    readonly ArtFeatureCache features;
    readonly MapMatcher matcher;
    readonly DoorLocator doors;
    readonly IslandTracker islandTracker;
    readonly HeadingDetector headingDetector;
    double? heading;          // degrees clockwise from north, smoothed
    DateTime headingTime;
    int outsideStreak;        // consecutive minimap reads that look like the island, not the Sanctum
    int insideStreak;         // ... and that look like the Sanctum
    // Outside the Sanctum: position on the island, and whether that's where the player is now.
    bool onIsland;
    Point2d? islandPlayer;
    DateTime islandTime, sanctumOkTime;
    MapPose? islandPose;
    long lastIslandSearch = -10000;
    int islandProbation;
    readonly FloorRefiner refiner;
    readonly MinimapTracker minimap;
    readonly GameWindow game = new();
    Thread? thread;
    volatile bool running;
    volatile bool resetRequested;
    bool toggled;

    public event Action<OverlayFrame>? Frame;
    public event Action? StatusChanged;

    // State shown in the tray tooltip and settings window.
    public bool GameFound { get; private set; }
    public OpenCvSharp.Rect GameClient => game.Client;
    public string? OcrMissingLanguage => ocr.MissingLanguage;
    /// <summary>The game's own language setting changed (with the game language on Auto).</summary>
    public event Action? LanguageDetected;
    /// <summary>A short notice for the player: text, and the difficulty it's about (for its colour;
    /// null for a Sanctum).</summary>
    public event Action<string, string?>? Notify;
    public string? DetectedDifficulty { get; private set; }
    public Layout? CurrentLayout { get; private set; }
    public string LayoutSource { get; private set; } = "";
    public string LastMatchInfo { get; private set; } = "";
    public string LastOcrText => difficulty.LastText;
    public string OcrLanguage => ocr.Language;
    public double MinimapScore => minimap.LastScore;
    public string MinimapInfo => minimap.LastInfo;

    public string? EffectiveDifficulty => settings.Difficulty == "Auto" ? DetectedDifficulty : settings.Difficulty;

    Point2d? player;
    DateTime playerTime;
    MapPose? lastPose;
    DateTime lastPoseTime;

    public Engine(LayoutDb db, Settings settings, KeyHook hook)
    {
        // OpenCV splits work over every core by default (16 here). The per-tick matches are tiny
        // and ran slower that way (heading 12 ms -> 7 ms on 1-4 threads), and bursts on all cores
        // take CPU from the game. Four keeps the background whole-map searches reasonably quick.
        Cv2.SetNumThreads(4);
        this.db = db;
        this.settings = settings;
        this.hook = hook;
        difficulty = new DifficultyDetector(ocr);
        arts = new MapArtCache(db);
        features = new ArtFeatureCache(arts);
        matcher = new MapMatcher(features);
        refiner = new FloorRefiner(arts);
        islandTracker = new IslandTracker(arts);
        headingDetector = new HeadingDetector(System.IO.Path.Combine(db.Root, "icons", "player_arrow.png"));
        string icons = System.IO.Path.Combine(db.Root, "icons");
        doors = new DoorLocator(arts, Cv2.ImRead(System.IO.Path.Combine(icons, "map_header.png")),
            Cv2.ImRead(System.IO.Path.Combine(icons, "screen_door_orange.png")), Cv2.ImRead(System.IO.Path.Combine(icons, "screen_door_blue.png")));
        minimap = new MinimapTracker(arts);
        hook.StateChanged += down => { if (down && settings.ToggleMode) toggled = !toggled; };
        if (!ocr.Available) Log.Write("Windows OCR is not available: difficulty must be set by hand");
        // The game's log says where the player is (island, Sanctum, elsewhere) and when a run
        // starts; read what it already holds so a run in progress is known from the start.
        gameLog.Changed += OnGameLogChanged;
        if (settings.UseGameLog) gameLog.Poll();
        bool sameRun = settings.RunId == null || gameLog.RunId == null || settings.RunId == gameLog.RunId;
        if (!sameRun) Log.Write("game log: a different run than the saved layout's, not restoring it");
        // Restarted mid-run: pick the run's layout back up (runs last at most 20 minutes).
        if (sameRun && settings.RunLayout != 0 && (DateTime.UtcNow - settings.RunLayoutTime).TotalMinutes < 25 && db.ByNumber(settings.RunLayout) is Layout saved)
        {
            CurrentLayout = saved;
            LayoutSource = "saved run";
            Log.Write($"layout restored: {saved.Number} (saved {(DateTime.UtcNow - settings.RunLayoutTime).TotalMinutes:0} min ago)");
        }
    }

    public void Start()
    {
        running = true;
        thread = new Thread(Loop) { IsBackground = true, Name = "SanctumHelper engine", Priority = ThreadPriority.BelowNormal };
        thread.Start();
        headingThread = new Thread(HeadingLoop) { IsBackground = true, Name = "SanctumHelper heading", Priority = ThreadPriority.BelowNormal };
        headingThread.Start();
        // Precompute map features off the hot path.
        ThreadPool.QueueUserWorkItem(_ =>
        {
            try { features.Prepare(db.Candidates(EffectiveDifficulty)); } catch (Exception e) { Log.Write("prepare: " + e); }
        });
    }

    /// <summary>Forget the current run's layout (new run, or a wrong lock).</summary>
    public void ResetRun()
    {
        resetRequested = true;
    }

    IReadOnlyList<Layout> AllCandidates()
    {
        if (settings.ForcedLayout != 0 && db.ByNumber(settings.ForcedLayout) is Layout forced) return [forced];
        var c = db.Candidates(EffectiveDifficulty);
        return c.Count > 0 ? c : db.Candidates(null);
    }

    IReadOnlyList<Layout> Candidates()
    {
        if (settings.ForcedLayout != 0 && db.ByNumber(settings.ForcedLayout) is Layout forced) return [forced];
        if (CurrentLayout != null) return [CurrentLayout];
        var c = db.Candidates(EffectiveDifficulty);
        return c.Count > 0 ? c : db.Candidates(null);
    }

    void Lock(Layout l, string source)
    {
        if (CurrentLayout?.Number == l.Number) return;
        if (CurrentLayout != null)
        {
            // Different layout = a new run: forget everything tied to the old one.
            lastPose = null; player = null; minimap.Reset();
            prevGray?.Dispose(); prevGray = null; prevGrayPose = null;
        }
        CurrentLayout = l;
        LayoutSource = source;
        misfitSince = null;
        // The layout says the difficulty when it belongs to just one: corrects a stale reading
        // (difficulty changed on the party screen just before starting).
        if (settings.Difficulty == "Auto" && l.Difficulties.Count == 1 && l.Difficulties[0] != DetectedDifficulty)
        {
            Log.Write($"difficulty: {l.Difficulties[0]} (from {l.Title}; was {DetectedDifficulty ?? "unknown"})");
            DetectedDifficulty = l.Difficulties[0];
            Notify?.Invoke(Lang.F("{0} Mode found.", Lang.Difficulty(l.Difficulties[0])), l.Difficulties[0]);
        }
        settings.RunLayout = l.Number;
        settings.RunLayoutTime = DateTime.UtcNow;
        settings.RunId = gameLog.RunId;
        try { settings.Save(); } catch (Exception e) { Log.Write("save run layout: " + e.Message); }
        Log.Write($"layout locked: {l.Number} via {source}");
        arts.Trim(n => n == l.Number); refiner.Trim(n => n == l.Number); minimap.Release(n => n == l.Number);
        Notify?.Invoke(Lang.F("Sanctum found #{0}", l.Number), null);
        StatusChanged?.Invoke();
    }

    void Loop()
    {
        var sw = clock;
        long nextWindow = 0, nextOcr = 0, nextMinimap = 0, nextGameLog = 0, nextLangCheck = 5000, nextPerfLog = 10000;
        bool wasShowing = false;
        while (running)
        {
            long now = sw.ElapsedMilliseconds;
            try
            {
                if (resetRequested)
                {
                    resetRequested = false;
                    CurrentLayout = null; lastPose = null; player = null; minimap.Reset();
                    onIsland = false; islandPlayer = null; islandPose = null; islandTracker.Reset();
                    ClearGuesses();
                    arts.Trim(_ => false); refiner.Trim(_ => false); minimap.Release();
                    settings.RunLayout = 0;
                    try { settings.Save(); } catch { /* not critical */ }
                    StatusChanged?.Invoke();
                }
                hook.Poll();
                // The controller button works like the overlay key (held, or pressed to toggle).
                bool padDown = false;
                if (settings.HoldButton != 0)
                {
                    pad.Poll();
                    padDown = pad.IsDown(settings.HoldButton);
                    if (padDown && !padWasDown && settings.ToggleMode) toggled = !toggled;
                }
                padWasDown = padDown;
                // The game's language can be changed in game: follow it when set to Auto.
                if (now >= nextLangCheck)
                {
                    nextLangCheck = now + 5000;
                    if (settings.GameLanguage == "Auto" && Lang.ReadGameLanguage() is string gl && gl != Lang.DetectedGameCode)
                    {
                        Log.Write($"game language changed to {gl}");
                        LanguageDetected?.Invoke();
                    }
                }
                if (now >= nextPerfLog)
                {
                    nextPerfLog = now + 10000;
                    var mem = System.Diagnostics.Process.GetCurrentProcess().PrivateMemorySize64 / (1 << 20);
                    Log.Write($"perf: track {perfTrack.Report()} · heading {perfHeading.Report()} · ocr {perfOcr.Report()} · overlay {OverlayView.Perf.Report()} · {mem} MB");
                }
                if (settings.UseGameLog && now >= nextGameLog)
                {
                    nextGameLog = now + 250;
                    gameLog.Poll();
                }
                if (now >= nextWindow)
                {
                    nextWindow = now + 1000;
                    bool found = game.Refresh(settings.WindowTitle);
                    if (found != GameFound)
                    {
                        GameFound = found;
                        Log.Write(found ? $"game window found: {game.Client.Width}x{game.Client.Height} at {game.Client.X},{game.Client.Y}" : "game window lost");
                        StatusChanged?.Invoke();
                    }
                }
                if (!GameFound)
                {
                    if (wasShowing) { Frame?.Invoke(new OverlayFrame(default, false, null, null, null, false, null, "")); wasShowing = false; }
                    Thread.Sleep(400);
                    continue;
                }

                bool show = (settings.ToggleMode ? toggled : hook.IsDown || padDown) && game.IsForeground;
                if (show && !wasShowing) { miniShowing = false; heldSince = now; Log.Write($"overlay key down ({(padDown ? "controller " + Gamepad.Name(settings.HoldButton) : KeyHook.Name(hook.Key))})"); }
                if (show)
                {
                    // One window capture per frame: it says whether the in-game map is open, and
                    // feeds either the full-map overlay or (map closed) the big zoom map.
                    var c0 = game.Client;
                    using (var frame = Grab(new Rect(0, 0, c0.Width, c0.Height)))
                    {
                        bool bigOk = !onIsland && (settings.ForcedLayout != 0 || CurrentLayout != null) && InRun;
                        if (!bigOk || MapScreenOpen(frame)) ShowMap(frame);
                        else { TrackMinimap(frame); ShowBig(); }
                    }
                    wasShowing = true;
                    Thread.Sleep(15);
                    continue;
                }
                if (wasShowing)
                {
                    Frame?.Invoke(new OverlayFrame(game.Client, false, CurrentLayout, null, null, false, EffectiveDifficulty, ""));
                    wasShowing = false;
                }
                // The party screen is read even with another window in front (e.g. this app's
                // settings, to see the difficulty come in); window capture doesn't need focus.
                // The party screen only exists outside a run: read it often there (a difficulty
                // changed just before starting is still caught) and never during one.
                bool inRunNow = InRun || GameLogPlace is GameLog.Place.Island or GameLog.Place.Sanctum;
                if (now >= nextOcr && settings.Difficulty == "Auto" && ocr.Available && !inRunNow)
                {
                    nextOcr = now + 500;
                    var t0 = perfClock.ElapsedTicks;
                    ReadPartyScreen();
                    perfOcr.Add(perfClock.ElapsedTicks - t0);
                }
                if (!game.IsForeground) { HideMini(); Thread.Sleep(250); continue; }
                if (now >= nextMinimap && settings.TrackMinimap)
                {
                    // Faster while the on-screen map follows the player; tracking is cheap then.
                    // On the island there may be no Sanctum layout yet; it still needs fast updates.
                    bool following = settings.MiniMapMode != "Off" && (CurrentLayout != null || onIsland);
                    // ~20 fixes a second while following: the shared corner grab is ~30 a second,
                    // and more fixes than that only re-track the same frame.
                    nextMinimap = now + (following ? 40 : 350);
                    var t1 = perfClock.ElapsedTicks;
                    TrackMinimap();
                    ShowMini();
                    perfTrack.Add(perfClock.ElapsedTicks - t1);
                }
                Thread.Sleep(15);
            }
            catch (Exception e)
            {
                Log.Write("engine: " + e);
                Thread.Sleep(500);
            }
        }
    }

    bool miniShowing;
    // ---- Facing and camera: read from the minimap on a loop of their own, ~30 times a second
    // (the tracking tick is ~10 Hz once it has done the position), so turning the camera shows
    // up at once; the overlay glides between readings at display rate.
    Thread? headingThread;

    void HeadingLoop()
    {
        var sw = Stopwatch.StartNew();
        while (running)
        {
            long t0 = sw.ElapsedMilliseconds;
            try
            {
                if (!GameFound || !game.IsForeground || !InRun) { Thread.Sleep(150); continue; }
                var c = game.Client;
                double ui = c.Height / 1080.0;
                var rc = Regions.ToClient(settings.MinimapRegion, Anchor.TopLeft, c.Width, c.Height);
                if (rc.Width < 40) { Thread.Sleep(250); continue; }
                // One copy of the HUD corner per cycle, for the arrow/cone here and for tracking
                // (each screen copy waits for the desktop to compose a frame: fewer is smoother).
                var corner = CornerRect(rc, c, ui);
                var top = ScreenGrabber.Grab(new Rect(c.X + corner.X, c.Y + corner.Y, corner.Width, corner.Height));
                lock (cornerGate) { sharedCorner?.Dispose(); sharedCorner = top; sharedCornerAt = Environment.TickCount64; }
                using (var mm = new Mat(top, rc))
                {
                    var t2 = perfClock.ElapsedTicks;
                    UpdateHeading(mm, ui);
                    perfHeading.Add(perfClock.ElapsedTicks - t2);
                }
            }
            catch (Exception e) { Log.Write("heading: " + e.Message); Thread.Sleep(500); }
            int left = 33 - (int)(sw.ElapsedMilliseconds - t0);
            if (left > 0) Thread.Sleep(left);
        }
    }

    readonly object cornerGate = new();
    Mat? sharedCorner;
    long sharedCornerAt;

    /// <summary>The HUD corner: the minimap, the Esc button and the run timer below them.</summary>
    static Rect CornerRect(Rect rc, Rect client, double ui) =>
        new(0, 0, Math.Max(rc.Right, (int)(250 * ui)), Math.Min(client.Height, Math.Max(rc.Bottom, (int)(650 * ui))));

    /// <summary>A copy of the heading loop's latest corner grab when it is fresh (null otherwise).</summary>
    Mat? TakeSharedCorner(Rect corner)
    {
        lock (cornerGate)
        {
            if (sharedCorner == null || Environment.TickCount64 - sharedCornerAt > 60 || sharedCorner.Width != corner.Width || sharedCorner.Height != corner.Height)
                return null;
            return sharedCorner.Clone();
        }
    }

    void UpdateHeading(Mat mm, double ui)
    {
        var now = DateTime.UtcNow;
        // Facing: the yellow arrow when it shows; otherwise (twined with an egg, the egg marker
        // replaces it) the white view cone, i.e. where the camera looks. While the arrow is
        // missing it is only retried twice a second, as its full search is the costly part.
        double? hint = heading is double hp && (now - headingTime).TotalSeconds < 0.5 && headingFromArrow ? hp : null;
        double? h = null;
        double weight = 0.6;
        if (now >= arrowRetry)
        {
            h = headingDetector.Detect(mm, ui, hint);
            arrowRetry = h == null ? now.AddMilliseconds(500) : default;
        }
        headingFromArrow = h != null;
        if (h != null) lastArrowSeen = now;
        // The view cone: where the camera looks. Drawn on our maps like the game does, and the
        // facing when the arrow is gone (smoothed more: a soft wedge is noisier than the arrow).
        // 0.2: on a recorded camera spin, good readings scored 0.22-0.30 when the cone lay over walls;
        // a 0.3 cut-off dropped ~40% of them and the cone froze, then leapt (31 big swings -> 2).
        if (HeadingDetector.FromViewCone(mm, ui, out double coneConf) is double ch && coneConf >= 0.2)
        {
            // Adaptive smoothing (one-euro filter): steady while the camera holds still, no lag while it turns.
            camera = (now - cameraTime).TotalSeconds < 1 ? cameraFilter.Next(ch, now) : cameraFilter.Reset(ch, now);
            cameraTime = now;
            // A blink of the arrow (particles) keeps the last heading; only a lasting absence means
            // egg form. (Camera and character can face different ways: switching would jump.)
            if (h == null && (now - lastArrowSeen).TotalSeconds > 0.6) { h = ch; weight = 0.45; }
        }
        if (h is double hv)
        {
            if (heading is double prev && (now - headingTime).TotalSeconds < 1)
            {
                double d = ((hv - prev) % 360 + 540) % 360 - 180;
                heading = (prev + weight * d + 360) % 360;
            }
            else heading = hv;
            headingTime = now;
        }
        // Twined with an egg: no arrow, and the egg marker right where the arrow would be (an egg
        // lying nearby sits off-centre with the arrow still showing). Measured: twined 0.75-0.85,
        // nearby egg <= 0.44 (arrow shown), normal minimaps <= 0.50.
        if (h == null || !headingFromArrow)
        {
            double e = (now - lastArrowSeen).TotalSeconds > 0.4 ? HeadingDetector.EggScore(mm, ui, iconsDir) : 0;
            eggVotes = e >= 0.6 ? Math.Min(eggVotes + 1, 3) : Math.Max(eggVotes - 1, 0);
        }
        else eggVotes = 0;
        if (eggVotes >= 2) lastEggSeen = now;
        // On as soon as two readings agree; off only after the egg has been gone for 0.6 s (a single
        // stray arrow reading used to flip it back and forth).
        bool egg = eggVotes >= 2 || (eggForm && (now - lastEggSeen).TotalSeconds < 0.6);
        if (egg != eggForm) Log.Write(egg ? "egg form: twined with an egg" : "egg form: back to normal");
        eggForm = egg;
        HeadingFeed.Publish(CurrentHeading, CurrentCamera, eggForm);
    }

    DateTime arrowRetry, lastArrowSeen, cameraTime;
    readonly AngleFilter cameraFilter = new();
    int eggVotes;
    bool eggForm;
    DateTime lastEggSeen;
    string iconsDir => System.IO.Path.Combine(db.Root, "icons");
    double? camera;
    double? CurrentCamera => camera is double c && (DateTime.UtcNow - cameraTime).TotalSeconds < 1.5 ? c : null;
    bool headingFromArrow;
    double? CurrentHeading => heading is double h && (DateTime.UtcNow - headingTime).TotalSeconds < 3 ? h : null;

    /// <summary>The on-screen map (corner or see-through) while the full map is closed. Only while
    /// the minimap keeps finding the player, so menus, the full map and loading screens hide it.</summary>
    void ShowMini()
    {
        var layout = settings.ForcedLayout != 0 ? db.ByNumber(settings.ForcedLayout) : CurrentLayout;
        var me = player;
        var when = playerTime;
        int floor = minimap.Floor;
        if (onIsland && settings.IslandMap && db.Island is Layout isl) { layout = isl; me = islandPlayer; when = islandTime; floor = 0; }
        // Outside the Sanctum (island minimap on screen) the Sanctum map is wrong: step aside.
        if (outsideStreak >= 3 && !settings.IslandMap) { HideMini(); return; }
        // While the game's HUD is up, keep showing the last fix (rooms with portal glow or heavy
        // effects can stall the minimap match for a while); hide it in menus and on the map.
        bool fresh = me != null && hudVisible && InRun && (DateTime.UtcNow - when).TotalSeconds < 60;
        if (settings.MiniMapMode == "Off" || layout == null || !fresh) { HideMini(); return; }
        miniShowing = true;
        Frame?.Invoke(new OverlayFrame(game.Client, true, layout, null, me, true, EffectiveDifficulty, "", Mini: true, Floor: floor, Heading: CurrentHeading,
            Velocity: onIsland ? islandTracker.Velocity : minimap.Velocity, Stamp: when, Camera: CurrentCamera));
    }

    /// <summary>Key held with the in-game map closed: a big map centred on the player, following
    /// them live (Sanctum only).</summary>
    void ShowBig()
    {
        var layout = settings.ForcedLayout != 0 ? db.ByNumber(settings.ForcedLayout) : CurrentLayout;
        bool fresh = player != null && (DateTime.UtcNow - playerTime).TotalSeconds < 60;
        if (layout == null || !fresh)
        {
            Frame?.Invoke(new OverlayFrame(game.Client, false, null, null, null, false, EffectiveDifficulty, ""));
            return;
        }
        Frame?.Invoke(new OverlayFrame(game.Client, true, layout, null, player, true, EffectiveDifficulty, "", Mini: true, Floor: minimap.Floor,
            Heading: CurrentHeading, Big: true, Velocity: minimap.Velocity, Stamp: playerTime, Camera: CurrentCamera));
    }

    void HideMini()
    {
        if (!miniShowing) return;
        miniShowing = false;
        Frame?.Invoke(new OverlayFrame(game.Client, false, null, null, null, false, null, ""));
    }

    bool printWindowFailed;
    int windowCaptureMisses;

    /// <summary>A part of the game's client area (client coordinates).</summary>
    Mat Grab(Rect sub)
    {
        var c = game.Client;
        if (settings.RecordableOverlay && !printWindowFailed)
        {
            var m = ScreenGrabber.GrabWindow(game.Handle, c.Width, c.Height, sub);
            if (m != null) { windowCaptureMisses = 0; return m; }
            // A single black frame is normal (fades, loading screens): use the screen for this
            // frame only. Only a long run of them means the window can't be captured at all
            // (e.g. exclusive fullscreen).
            if (++windowCaptureMisses >= 40)
            {
                printWindowFailed = true;
                Log.Write("window capture keeps giving no image; using screen capture (overlay hidden from recordings)");
                CaptureFallback?.Invoke();
            }
        }
        return ScreenGrabber.Grab(new Rect(c.X + sub.X, c.Y + sub.Y, sub.Width, sub.Height));
    }

    /// <summary>Raised when window capture fails, so the overlay can hide itself from captures again.</summary>
    public event Action? CaptureFallback;
    public bool UsingWindowCapture => settings.RecordableOverlay && !printWindowFailed;

    void ReadPartyScreen()
    {
        var c = game.Client;
        var rc = Regions.ToClient(settings.PartyGoalRegion, Anchor.TopCenter, c.Width, c.Height);
        // Grab only a band around the goal line; cheap enough to do every 1.5 s.
        using var band = Grab(new Rect(0, 0, c.Width, Math.Min(c.Height, rc.Bottom + 10)));
        var d = difficulty.Detect(band, settings.PartyGoalRegion, c.Width, c.Height);
        if (d == null) return;
        // Being on the party screen means any previous run is over.
        if (CurrentLayout != null && settings.ForcedLayout == 0) ResetRun();
        difficultyDoubted = false;
        if (d != DetectedDifficulty)
        {
            DetectedDifficulty = d;
            Log.Write($"difficulty: {d} (\"{difficulty.LastText}\")");
            Notify?.Invoke(Lang.F("{0} Mode found.", Lang.Difficulty(d)), d);
            StatusChanged?.Invoke();
            ThreadPool.QueueUserWorkItem(_ => { try { features.Prepare(db.Candidates(d)); } catch { } });
        }
    }

    /// <summary>Is the in-game map screen showing? (Its title icon, top-left.)</summary>
    bool MapScreenOpen(Mat frame)
    {
        double f = 1080.0 / frame.Height;
        var area = new Rect(0, 0, Math.Min(frame.Width, (int)(400 / f)), Math.Min(frame.Height, (int)(200 / f)));
        using var crop = new Mat(frame, area);
        using var small = new Mat();
        Cv2.Resize(crop, small, new Size(), f, f, InterpolationFlags.Area);
        return doors.IsMapOpen(small);
    }

    void TrackMinimap(Mat? fullFrame = null)
    {
        // The minimap alone can't tell layouts apart at the start (only the spawn room is
        // revealed), so it tracks the player once the full map has identified the layout.
        var layout = settings.ForcedLayout != 0 ? db.ByNumber(settings.ForcedLayout) : CurrentLayout;
        var c = game.Client;
        double ui = c.Height / 1080.0;
        var rc = Regions.ToClient(settings.MinimapRegion, Anchor.TopLeft, c.Width, c.Height);
        if (rc.Width < 40) return;
        // One grab for the HUD corner: minimap, the Esc button, and the run timer below them.
        var corner = CornerRect(rc, c, ui);
        // From the frame we already have, or a copy of just that corner from the screen (13 ms vs
        // 27 ms for a whole-window capture; no overlay ever draws over the HUD corner).
        using var top = fullFrame != null ? new Mat(fullFrame, corner).Clone()
            : TakeSharedCorner(corner) ?? ScreenGrabber.Grab(new Rect(c.X + corner.X, c.Y + corner.Y, corner.Width, corner.Height));
        UpdateInRun(top, ui);
        // The HUD is up if either the Esc button or the run timer shows (two separate elements:
        // particles, a dash or 4K scaling can spoil one for a moment). Only treat it as gone
        // after half a second without either, so the on-screen map doesn't blink.
        if (lastTimerSeen || HudVisible(top, ui)) { hudLastSeen = DateTime.UtcNow; hudVisible = true; }
        else if ((DateTime.UtcNow - hudLastSeen).TotalSeconds > 0.5) hudVisible = false;
        if (!InRun) return;
        // A chest, shop or other screen covering the HUD (timer match ~0.27 under it, vs 0.65+
        // for the usual flicker): the minimap is hidden, so skip it instead of counting misses.
        if (lastTimerScore < 0.5 && !timerTemplate!.Empty()) return;
        using var mm = new Mat(top, rc);
        var now = DateTime.UtcNow;
        bool logNow = clock.ElapsedMilliseconds - lastMinimapLog > 3000;
        if (logNow) lastMinimapLog = clock.ElapsedMilliseconds;
        // Inside the Sanctum or out on the island? The Sanctum minimap always shows dark void and
        // walls (12-60% of the disc, measured); the island minimap is almost all map art (~1%).
        // The game's log knows (it logs each area as loading starts); the minimap's look is the
        // fallback when the log isn't there.
        bool islandLike = settings.UseGameLog && gameLog.Where is GameLog.Place.Island or GameLog.Place.Sanctum
            ? gameLog.Where == GameLog.Place.Island
            : LooksOutside(mm);
        // Back inside: a few Sanctum-looking minimaps in a row end island mode, whether or not a
        // Sanctum layout is known yet (a run can start outside).
        insideStreak = islandLike ? 0 : insideStreak + 1;
        if (onIsland && insideStreak >= 3)
        {
            onIsland = false;
            islandTracker.Reset();
            Log.Write("back in the Sanctum (minimap)");
        }
        // A run can start on the island, before any Sanctum layout is known: track there anyway.
        if (layout == null && !(islandLike && settings.IslandMap))
        {
            // The layout comes from the minimap as standard; the full map (doors) is the fallback
            // and overrides a guess it clearly disagrees with.
            if (hudVisible && !islandLike) GuessLayout(mm, ui, logNow);
            else if (islandLike && outsideStreak > 20 && guesses.Count > 0 && guessTask == null) { ClearGuesses(); Log.Write("guess: reset (left the Sanctum)"); }
            return;
        }
        // That decides which map the minimap is matched against: the island minimap is mostly
        // light grey and can pass as Sanctum floor, and Sanctum views can near-tie on the island.
        outsideStreak = islandLike ? outsideStreak + 1 : 0;
        bool trySanctum = !islandLike && layout != null;

        if (trySanctum)
        {
            var pos = minimap.Locate(mm, layout!, ui);
            if (logNow) Log.Write($"{minimap.LastInfo} -> {(pos == null ? "rejected" : "ok")} ({layout!.Title})");
            // A wrong layout (e.g. locked from a stale difficulty) never matches. Only frames that
            // show plenty of floor count, so menus, chests and corridor glimpses don't.
            if (pos != null || !hudVisible || minimap.LastInfo.StartsWith("minimap: too little")) { if (pos != null) misfitSince = null; }
            else misfitSince ??= now;
            if (misfitSince is DateTime ms && (now - ms).TotalSeconds > 30 && settings.ForcedLayout == 0 && CurrentLayout != null)
            {
                Log.Write($"{CurrentLayout.Title} hasn't matched the minimap for 30 s: identifying the layout again, every difficulty");
                misfitSince = null;
                CurrentLayout = null; LayoutSource = ""; lastPose = null; player = null; minimap.Reset();
                settings.RunLayout = 0;
                difficultyDoubted = true;
                ClearGuesses();
                StatusChanged?.Invoke();
                return;
            }
            if (logNow && settings.SaveDebugFrames) SaveDebug(mm, pos == null ? "minimap_rejected" : "minimap_ok");
            if (pos != null)
            {
                if (onIsland) Log.Write("back in the Sanctum");
                onIsland = false;
                player = pos;
                playerTime = minimapTime = sanctumOkTime = now;
                return;
            }
        }
        if (!settings.IslandMap || db.Island is not Layout isl || !hudVisible || !islandLike) { odoPrev?.Dispose(); odoPrev = null; return; }
        // How far the player moved since the last frame, from how far the minimap scrolled. It
        // carries the position across open ground where nothing on the minimap can be matched.
        var step = MinimapStep(mm, ui);
        Point2d? dead = onIsland && islandPlayer is Point2d cur0 && (now - islandTime).TotalSeconds < 3
            ? new Point2d(cur0.X + (step?.X ?? 0), cur0.Y + (step?.Y ?? 0)) : null;
        // A Sanctum portal icon in view pins the position exactly, even on open ground where the
        // minimap shows nothing to match (big grass fields): it overrides a drifting image match.
        if (islandSearch == null && IslandLandmark(mm, ui, isl) is Point2d lm)
        {
            // Whenever the portal is in view it decides (the image match drifts on open ground);
            // the tracker restarts from there so it follows on once the portal leaves the view.
            if (logNow || !onIsland) Log.Write($"island: placed by the portal icon at ({lm.X:0},{lm.Y:0})");
            islandTracker.Reset();
            islandTracker.Seed(lm);
            islandArrival = null;
            islandProbation = 0;
            onIsland = true;
            islandPlayer = lm;
            islandTime = now;
            lastLandmark = lm;
            lastLandmarkAt = now;
            return;
        }
        if (islandArrival != null && islandSearch == null)
        {
            Point2d? best = null;
            double bestScore = 0;
            foreach (var m in isl.Markers.Where(m => m.Kind == islandArrival))
            {
                islandTracker.Reset();
                var spot = new Point2d(m.Img[0], m.Img[1]);
                islandTracker.Seed(spot);
                if (islandTracker.Locate(mm, isl, ui, false) is Point2d && islandTracker.LastScore > bestScore)
                {
                    // The player stands at the portal / landing itself: start there, not at the
                    // (weaker, possibly off) match that picked it.
                    best = spot;
                    bestScore = islandTracker.LastScore;
                }
            }
            islandTracker.Reset();
            if (best is Point2d at)
            {
                islandTracker.Seed(at);
                Log.Write($"island: found at the {(islandArrival == "boat" ? "boat landing" : "blue-door exit")} ({bestScore:0.00})");
                islandArrival = null;
                islandProbation = 0;
                onIsland = true;
                islandPlayer = at;
                islandTime = now;
                return;
            }
            // Not there (yet): loading, or already walked off. Fall back to searching after ~3 s.
            if (++islandArrivalTries > 40) islandArrival = null;
            return;
        }
        // A whole-island search takes a second or two: it runs in the background so tracking, the
        // arrow and the overlays keep updating meanwhile. Its result is picked up first thing,
        // whatever the tracking state (a hit makes the tracker "tracking" before the task ends,
        // which used to strand the result for 20 s).
        Point2d? ip;
        bool fromSearch = false;
        if (islandSearch != null)
        {
            if (!islandSearch.IsCompleted) return;
            ip = islandSearch.IsCompletedSuccessfully ? islandSearch.Result : null;
            islandSearch = null;
            fromSearch = logNow = true;
        }
        else if (islandTracker.Tracking) ip = islandTracker.Locate(mm, isl, ui, false);
        else
        {
            if (clock.ElapsedMilliseconds - lastIslandSearch < 1200) return;
            lastIslandSearch = clock.ElapsedMilliseconds;
            var shot = mm.Clone();
            islandSearch = System.Threading.Tasks.Task.Run(() =>
            {
                try { return islandTracker.Locate(shot, isl, ui, true); }
                finally { shot.Dispose(); }
            });
            return;
        }
        if (logNow) Log.Write($"{islandTracker.LastInfo} -> {(ip == null ? "rejected" : "ok")}");
        if (ip == null)
        {
            if (islandProbation > 0) { islandProbation = 0; Log.Write("island candidate dropped (local tracking failed)"); }
            if (dead is Point2d d0 && step != null) MoveOnIsland(d0, now);
            return;
        }
        // Just placed by a portal icon: an image match far from there is a look-alike, not a move.
        if ((now - lastLandmarkAt).TotalSeconds < 2 && Math.Abs(ip.Value.X - lastLandmark.X) + Math.Abs(ip.Value.Y - lastLandmark.Y) > 60)
        {
            islandTracker.Reset();
            islandTracker.Seed(lastLandmark);
            return;
        }
        if (!onIsland)
        {
            // A first whole-island hit is confirmed by tracking on from it: the next few quick
            // local fixes must also succeed (works while running; a wrong hit fails at once).
            if (fromSearch) { islandProbation = 4; Log.Write("island candidate: confirming by local tracking"); return; }
            if (islandProbation > 1) { islandProbation--; Log.Write($"island candidate: local fix {islandTracker.LastInfo}"); return; }
            islandProbation = 0;
        }
        if (onIsland && dead is Point2d d1)
        {
            double score = islandTracker.LastScore;
            bool agrees = Math.Abs(ip.Value.X - d1.X) + Math.Abs(ip.Value.Y - d1.Y) < 80;
            if (!(score >= 0.75 || (score >= 0.5 && agrees)))
            {
                // A weak match (open ground, near-ties) that disagrees with how the player moved
                // is a look-alike: follow the movement instead.
                MoveOnIsland(d1, now);
                return;
            }
        }
        if (!onIsland) Log.Write("on the island");
        onIsland = true;
        islandPlayer = ip;
        islandTime = now;
    }

    void MoveOnIsland(Point2d at, DateTime now)
    {
        islandTracker.Reset();
        islandTracker.Seed(at);
        islandPlayer = at;
        islandTime = now;
    }

    Mat? odoPrev;

    /// <summary>The player's movement since the last call (island map px), from the minimap's
    /// scroll between frames (phase correlation of its middle; the player marker and the white view
    /// cone, which turn with the camera, are blanked). Null when it can't tell.</summary>
    internal Point2d? MinimapStep(Mat mm, double ui)
    {
        using var small = new Mat();
        Cv2.Resize(mm, small, new Size(), 1 / ui, 1 / ui, InterpolationFlags.Area);
        int side = 128;
        var c = new Point(small.Width / 2, small.Height / 2);
        var roi = new Rect(c.X - side / 2, c.Y - side / 2, side, side).Intersect(new Rect(0, 0, small.Width, small.Height));
        if (roi.Width < side || roi.Height < side) return null;
        using var part = new Mat(small, roi);
        using var hsv = new Mat();
        Cv2.CvtColor(part, hsv, ColorConversionCodes.BGR2HSV);
        using var blank = new Mat();
        Cv2.InRange(hsv, new Scalar(0, 0, 201), new Scalar(180, 44, 255), blank);       // the white cone
        Cv2.Circle(blank, new Point(side / 2, side / 2), 18, Scalar.White, -1);         // the player marker
        using var gray = new Mat();
        Cv2.CvtColor(part, gray, ColorConversionCodes.BGR2GRAY);
        var cur = new Mat();
        gray.ConvertTo(cur, MatType.CV_32F);
        using (var inv = new Mat())
        {
            Cv2.BitwiseNot(blank, inv);
            var mean = Cv2.Mean(cur, inv);
            cur.SetTo(mean, blank);
        }
        var prev = odoPrev;
        odoPrev = cur;
        if (prev == null) return null;
        using (prev)
        {
            using var win = new Mat();
            Cv2.CreateHanningWindow(win, cur.Size(), MatType.CV_32F);
            var shift = Cv2.PhaseCorrelate(prev, cur, win, out double response);
            if (response < 0.08 || Math.Abs(shift.X) + Math.Abs(shift.Y) > 24) return null;
            // The map scrolls the opposite way to the player; 1.2 minimap px per island map px.
            return new Point2d(-shift.X / 1.2, -shift.Y / 1.2);
        }
    }

    readonly GameLog gameLog = new();
    /// <summary>Where the game's log says the player is (Unknown without it).</summary>
    public GameLog.Place GameLogPlace => settings.UseGameLog ? gameLog.Where : GameLog.Place.Unknown;

    void OnGameLogChanged(GameLog.Place where, bool newRun)
    {
        Log.Write($"game log: {where}{(newRun ? " (new run)" : "")}");
        // A new run: whatever layout the last one had is wrong now.
        if (newRun && settings.ForcedLayout == 0 && CurrentLayout != null && settings.RunId != gameLog.RunId)
        {
            Log.Write("game log: new run, forgetting the last layout");
            ResetRun();
        }
        switch (where)
        {
            case GameLog.Place.Sanctum when onIsland:
                // Into the Sanctum: drop the island map at once instead of after a few minimaps.
                onIsland = false;
                islandTracker.Reset();
                Log.Write("back in the Sanctum (game log)");
                break;
            case GameLog.Place.Island:
                // Out on the island: the Sanctum map goes, island tracking starts searching now.
                insideStreak = 0;
                outsideStreak = Math.Max(outsideStreak, 3);
                // The player arrives somewhere known: out of the blue door at one of the Sanctum's
                // exit portals, or at a run's start at one of the boat landings. Try those spots
                // first; a whole-island search can't place open ground (the grass fields).
                islandArrival = newRun ? "boat" : "entrance_side";
                islandArrivalTries = 0;
                break;
        }
        StatusChanged?.Invoke();
    }

    long lastMinimapLog = -10000;
    DateTime? misfitSince;
    string? islandArrival;
    int islandArrivalTries;
    Mat? portalIcon;
    Point2d lastLandmark;
    DateTime lastLandmarkAt;

    /// <summary>The player's island position from a Sanctum exit-portal icon on the minimap: its
    /// offset from the centre (1.2 minimap px per map px) back from that portal's place. Of the
    /// five portals, the one that fits the last known position (they are 800+ px apart).</summary>
    Point2d? IslandLandmark(Mat mm, double ui, Layout isl)
    {
        Point2d? near = islandPlayer is Point2d ip0 && (DateTime.UtcNow - islandTime).TotalSeconds < 30 ? ip0 : null;
        portalIcon ??= Cv2.ImRead(System.IO.Path.Combine(db.Root, "icons", "mm_portal_blue.png"));
        if (portalIcon.Empty()) return null;
        using var small = new Mat();
        Cv2.Resize(mm, small, new Size(), 1 / ui, 1 / ui, InterpolationFlags.Area);
        if (small.Width <= portalIcon.Width || small.Height <= portalIcon.Height) return null;
        using var res = new Mat();
        Cv2.MatchTemplate(small, portalIcon, res, TemplateMatchModes.CCoeffNormed);
        Cv2.MinMaxLoc(res, out _, out double max, out _, out Point loc);
        if (max < 0.75) return null;
        double dx = loc.X + portalIcon.Width / 2.0 - small.Width / 2.0, dy = loc.Y + portalIcon.Height / 2.0 - small.Height / 2.0;
        // Icons of far-off places are pinned to the rim (measured ~88 px out at 1080p); ones in
        // view sit inside it (a real portal measured at 81 px).
        if (Math.Sqrt(dx * dx + dy * dy) > 85 || Math.Sqrt(dx * dx + dy * dy) < 18) return null;
        var cands = isl.Markers.Where(m => m.Kind == "entrance_side").Select(m => new Point2d(m.Img[0] - dx / 1.2, m.Img[1] - dy / 1.2)).ToList();
        if (near is Point2d n)
        {
            Point2d? best = null;
            double bestD = 450;
            foreach (var p in cands)
            {
                double d = Math.Abs(p.X - n.X) + Math.Abs(p.Y - n.Y);
                if (d < bestD) { bestD = d; best = p; }
            }
            return best;
        }
        // No recent position (just started, or lost): which portal it is, the surroundings say —
        // a short local image match at each candidate, the clearly best one wins.
        var scored = new List<(Point2d p, double s)>();
        foreach (var p in cands)
        {
            islandTracker.Reset();
            islandTracker.Seed(p);
            islandTracker.Locate(mm, isl, ui, false);
            scored.Add((p, islandTracker.LastScore));
        }
        islandTracker.Reset();
        var ranked = scored.OrderByDescending(x => x.s).ToList();
        if (ranked.Count == 0 || ranked[0].s < 0.3 || (ranked.Count > 1 && ranked[0].s - ranked[1].s < 0.05)) return null;
        return ranked[0].p;
    }
    readonly Gamepad pad = new();
    // Timings for the "perf:" log line (every 10 s): how long and how often each part runs.
    static readonly Stopwatch perfClock = Stopwatch.StartNew();
    readonly PerfStat perfTrack = new(), perfHeading = new(), perfOcr = new();
    bool padWasDown;
    System.Threading.Tasks.Task<Point2d?>? islandSearch;

    // ---- Are we in an Egg Heist run? The run timer (stopwatch icon left of the countdown) is
    // on the HUD only during a run: measured 0.95-1.0 in runs, <0.35 in the open world and hub.
    Mat? timerTemplate;
    (Mat tpl, Mat mask)? redTimer;
    DateTime lastRunSeen, lastNoRunCheck = DateTime.UtcNow;
    // Held for 15 s: over bright grass or flowers the timer match dips for seconds at a time on
    // the island, and every dip used to pause tracking (and the island hit's confirmation).
    public bool InRun => (DateTime.UtcNow - lastRunSeen).TotalSeconds < 15;
    double lastTimerScore;
    bool lastTimerSeen;

    void UpdateInRun(Mat topLeft, double ui)
    {
        // Compared on brightness only (HSV value): the timer turns red under 3 minutes, and red
        // and white are the same brightness (measured: red 0.56-0.77 in colour, 0.92-0.99 here).
        timerTemplate ??= ValueChannel(Cv2.ImRead(System.IO.Path.Combine(db.Root, "icons", "hud_timer.png")));
        if (redTimer == null)
        {
            var tpl = Cv2.ImRead(System.IO.Path.Combine(db.Root, "icons", "hud_timer_red.png"));
            var mask = new Mat(tpl.Size(), MatType.CV_8U, Scalar.Black);
            if (!tpl.Empty()) Cv2.Circle(mask, new Point(tpl.Width / 2, tpl.Height / 2), tpl.Width / 2 - 2, Scalar.White, -1);
            redTimer = (tpl, mask);
        }
        var now = DateTime.UtcNow;
        bool seen = true;
        if (!timerTemplate.Empty())
        {
            using var small = new Mat();
            Cv2.Resize(topLeft, small, new Size(), 1 / ui, 1 / ui, InterpolationFlags.Area);
            var area = new Rect(15, 340, 140, 320).Intersect(new Rect(0, 0, small.Width, small.Height));
            seen = false;
            if (area.Width > timerTemplate.Width && area.Height > timerTemplate.Height)
            {
                using var res = new Mat();
                using var v = ValueChannel(new Mat(small, area));
                Cv2.MatchTemplate(v, timerTemplate, res, TemplateMatchModes.CCoeffNormed);
                Cv2.MinMaxLoc(res, out _, out double max, out _, out _);
                // Under 3 minutes the stopwatch becomes a red disc with a white running figure:
                // same brightness inside, so that one is matched in colour, inside its circle.
                if (max < 0.7 && redTimer is { } red && !red.tpl.Empty())
                {
                    using var area2 = new Mat(small, area);
                    using var r2 = new Mat();
                    Cv2.MatchTemplate(area2, red.tpl, r2, TemplateMatchModes.CCoeffNormed, red.mask);
                    Cv2.PatchNaNs(r2, -1);
                    Cv2.Threshold(r2, r2, 1.0, 0, ThresholdTypes.TozeroInv);
                    Cv2.MinMaxLoc(r2, out _, out double redMax, out _, out _);
                    max = Math.Max(max, redMax);
                }
                lastTimerScore = max;
                // seen at 0.7, lost below 0.55: scores hover near 0.7 over bright floor, which flipped it
                // back and forth (151 times in one evening's log)
                seen = max >= (lastTimerSeen ? 0.55 : 0.7);
            }
        }
        if (seen != lastTimerSeen)
        {
            lastTimerSeen = seen;
            Log.Write($"run timer {(seen ? "seen" : "lost")} ({lastTimerScore:0.00})");
        }
        // Tracking the player on the island means we're in a run, timer or not.
        if (!seen && onIsland && islandTracker.Tracking) seen = true;
        if (seen)
        {
            if ((now - lastRunSeen).TotalSeconds > 30) Log.Write("Egg Heist run detected (timer on screen)");
            lastRunSeen = now;
        }
        // Gone for 90 s with the game in front: the run is over, forget its layout. (The full
        // map and menus hide the timer briefly; nobody stays in them that long.)
        else if (CurrentLayout != null && settings.ForcedLayout == 0 && (now - lastRunSeen).TotalSeconds > 90 && (now - lastNoRunCheck).TotalSeconds > 5)
        {
            lastNoRunCheck = now;
            Log.Write("no Egg Heist run on screen for 90 s: forgetting the layout");
            ResetRun();
        }
        if (!InRun) HideMini();
    }

    static Mat ValueChannel(Mat bgr)
    {
        if (bgr.Empty()) return bgr;
        using var hsv = new Mat();
        Cv2.CvtColor(bgr, hsv, ColorConversionCodes.BGR2HSV);
        return hsv.ExtractChannel(2);
    }

    /// <summary>Does the minimap show the island rather than the Sanctum? The Sanctum minimap
    /// always has dark void and walls (12-100% of the disc, HSV value < 90) and never green;
    /// the island minimap has almost no dark (1-2%) and is often green (58% in rain).</summary>
    static bool LooksOutside(Mat minimapBgr)
    {
        using var hsv = new Mat();
        Cv2.CvtColor(minimapBgr, hsv, ColorConversionCodes.BGR2HSV);
        using var disc = new Mat(hsv.Size(), MatType.CV_8U, Scalar.Black);
        var c = new Point(hsv.Width / 2, hsv.Height / 2);
        int r = (int)(Math.Min(hsv.Width, hsv.Height) * 0.42);
        Cv2.Circle(disc, c, r, Scalar.White, -1);
        Cv2.Circle(disc, c, r / 5, Scalar.Black, -1);
        double n = Math.Max(1, Cv2.CountNonZero(disc));
        using var dark = new Mat();
        Cv2.InRange(hsv, new Scalar(0, 0, 0), new Scalar(180, 255, 89), dark);
        Cv2.BitwiseAnd(dark, disc, dark);
        using var green = new Mat();
        Cv2.InRange(hsv, new Scalar(30, 40, 90), new Scalar(95, 255, 255), green);
        Cv2.BitwiseAnd(green, disc, green);
        return Cv2.CountNonZero(dark) / n < 0.06 || Cv2.CountNonZero(green) / n > 0.25;
    }

    // ---- The layout from the minimap (standard; the full map is the fallback). Every run
    // starts at the orange door, so each candidate layout gets a tracker seeded at its orange
    // door; the right one keeps matching as the player explores, wrong ones don't. Rounds run in
    // the background: with the difficulty unknown that is up to 30 trackers per round.
    readonly Dictionary<int, (MinimapTracker t, List<double> scores)> guesses = new();
    System.Threading.Tasks.Task<(int won, int[] prune)>? guessTask;
    long nextGuess, guessStarted;
    bool wholeSearch, guessWidened;
    /// <summary>The minimap found that none of the detected difficulty's layouts fits: the
    /// difficulty reading is probably stale, so it no longer decides door ties.</summary>
    bool difficultyDoubted;
    /// <summary>True while the layout is being worked out from the minimap.</summary>
    public bool IdentifyingLayout => CurrentLayout == null && guesses.Count > 0;

    void ClearGuesses()
    {
        bool had = guesses.Count > 0;
        foreach (var g in guesses.Values) g.t.Release();   // each candidate's tracker holds its own scaled floors
        guesses.Clear();
        guessTask = null;
        if (had) StatusChanged?.Invoke();
    }

    void GuessLayout(Mat mm, double ui, bool logNow)
    {
        if (guessTask != null)
        {
            if (!guessTask.IsCompleted) return;
            var (won, prune) = guessTask.IsCompletedSuccessfully ? guessTask.Result : (0, []);
            if (!guessTask.IsCompletedSuccessfully) Log.Write("guess: " + guessTask.Exception?.InnerException?.Message);
            guessTask = null;
            foreach (var n in prune) if (guesses.Remove(n, out var gone)) gone.t.Release();
            if (won != 0 && CurrentLayout == null && db.ByNumber(won) is Layout winner)
            {
                var spot = guesses.TryGetValue(won, out var wg) ? wg.t.LastPosition : null;
                Log.Write($"guess: {winner.Title} after {(clock.ElapsedMilliseconds - guessStarted) / 1000.0:0.0} s");
                Lock(winner, "minimap");
                // Hand over to the main tracker from where the guess has the player now.
                minimap.Reset();
                minimap.Seed(spot ?? new Point2d(winner.Orange.Img[0], winner.Orange.Img[1]));
                ClearGuesses();
                return;
            }
        }
        if (clock.ElapsedMilliseconds < nextGuess) return;
        nextGuess = clock.ElapsedMilliseconds + 600;
        var cands = db.Candidates(EffectiveDifficulty);
        if (cands.Count == 0) cands = db.Candidates(null);
        if (guesses.Count == 0)
        {
            guessStarted = clock.ElapsedMilliseconds;
            wholeSearch = guessWidened = false;
            Log.Write($"guess: identifying the layout from the minimap ({cands.Count} candidates)");
            foreach (var l in cands)
            {
                var t = new MinimapTracker(arts);
                t.Seed(new Point2d(l.Orange.Img[0], l.Orange.Img[1]));
                guesses[l.Number] = (t, new List<double>());
            }
            StatusChanged?.Invoke();
        }
        // None of this difficulty's layouts fits after 10 s: the difficulty reading may be stale
        // (changed on the party screen just before starting). Add every other layout.
        if (!guessWidened && cands.Count < db.Candidates(null).Count && clock.ElapsedMilliseconds - guessStarted > 10000
            && guesses.Values.All(g => g.scores.Count >= 4 && g.scores.Average() < 0.45))
        {
            guessWidened = difficultyDoubted = true;
            Log.Write($"guess: no {EffectiveDifficulty} layout fits, trying every layout");
            foreach (var l in db.Candidates(null).Where(l => !guesses.ContainsKey(l.Number)))
            {
                var t = new MinimapTracker(arts);
                t.Seed(new Point2d(l.Orange.Img[0], l.Orange.Img[1]));
                guesses[l.Number] = (t, new List<double>());
            }
        }
        if (difficultyDoubted && !guessWidened)
        {
            guessWidened = true;
            foreach (var l in db.Candidates(null).Where(l => !guesses.ContainsKey(l.Number)))
            {
                var t = new MinimapTracker(arts);
                t.Seed(new Point2d(l.Orange.Img[0], l.Orange.Img[1]));
                guesses[l.Number] = (t, new List<double>());
            }
        }
        if (guessWidened) cands = db.Candidates(null);
        var live = guesses.Where(kv => cands.Any(c => c.Number == kv.Key))
            .Select(kv => (n: kv.Key, layout: db.ByNumber(kv.Key)!, t: kv.Value.t, scores: kv.Value.scores)).ToList();
        if (live.Count == 0) { ClearGuesses(); return; } // difficulty changed: start over next round
        // Nobody fits after 20 s: measured, the right layout averages 0.55+ within a few rounds
        // from the orange door; lost trackers sit around 0.3-0.4 on look-alike corridors.
        bool restart = !wholeSearch && clock.ElapsedMilliseconds - guessStarted > 20000
            && live.All(g => g.scores.Count >= 4 && g.scores.Average() < 0.45);
        if (restart) { guessStarted = clock.ElapsedMilliseconds; wholeSearch = true; }
        // Whole-layout searches cost ~0.25 s per layout (measured 7-8 s for all 30): slower rounds.
        if (wholeSearch) nextGuess = clock.ElapsedMilliseconds + 2500;
        var shot = mm.Clone();
        guessTask = System.Threading.Tasks.Task.Run(() =>
        {
            try
            {
                if (restart)
                {
                    // Nobody fits from the orange door: the player had left the spawn room before
                    // tracking began. Let every tracker search its whole layout instead.
                    foreach (var g in live) { g.t.Reset(); g.scores.Clear(); }
                    Log.Write("guess: no layout fits from the orange door, searching whole layouts");
                }
                foreach (var g in live)
                {
                    g.t.Locate(shot, g.layout, ui);
                    // Nothing to compare on this frame (menu, a glimpse of corridor): it says
                    // nothing about the layout, so it doesn't count for anyone.
                    if (g.t.LastInfo.StartsWith("minimap: too little")) return (0, Array.Empty<int>());
                    g.scores.Add(g.t.LastScore);
                    if (g.scores.Count > 8) g.scores.RemoveAt(0);
                }
                var ranked = live.Where(g => g.scores.Count >= 4)
                    .Select(g => (g.n, avg: g.scores.Average())).OrderByDescending(x => x.avg).ToList();
                if (ranked.Count == 0) return (0, Array.Empty<int>());
                if (logNow) Log.Write("guess: " + string.Join("  ", ranked.Take(4).Select(x => $"{x.n}:{x.avg:0.00}")));
                // Drop layouts that clearly don't fit, so rounds stay cheap with 30 candidates.
                var prune = ranked.Count > 3
                    ? ranked.Skip(3).Where(x => x.avg < 0.25 && ranked[0].avg - x.avg > 0.25).Select(x => x.n).ToArray()
                    : Array.Empty<int>();
                double second = ranked.Count > 1 ? ranked[1].avg : 0;
                return (ranked[0].avg >= 0.55 && ranked[0].avg - second >= 0.1 ? ranked[0].n : 0, prune);
            }
            finally { shot.Dispose(); }
        });
    }
    bool hudVisible;
    DateTime hudLastSeen;
    Mat? hudTemplate;

    /// <summary>Normal play (not a menu, the full map or a loading screen): the minimap's Esc
    /// button is on screen. Measured 0.96-1.0 in play, 0.28 on the map screen.</summary>
    bool HudVisible(Mat topLeft, double ui)
    {
        hudTemplate ??= Cv2.ImRead(System.IO.Path.Combine(db.Root, "icons", "hud_esc.png"));
        if (hudTemplate.Empty()) return true;
        using var small = new Mat();
        Cv2.Resize(topLeft, small, new Size(), 1 / ui, 1 / ui, InterpolationFlags.Area);
        if (small.Width < hudTemplate.Width || small.Height < hudTemplate.Height) return true;
        using var res = new Mat();
        Cv2.MatchTemplate(small, hudTemplate, res, TemplateMatchModes.CCoeffNormed);
        Cv2.MinMaxLoc(res, out _, out double max, out _, out _);
        return max >= 0.7;
    }
    DateTime minimapTime;
    Mat? prevGray;      // last aligned map frame, small grayscale, for following pans
    MapPose? prevGrayPose;
    const double PanScale = 0.25;

    static Mat PanImage(Mat frame)
    {
        // Centre of the screen only: the map's title, zoom bar and edges don't move with a pan.
        var roi = new Rect(frame.Width / 8, frame.Height / 8, frame.Width * 3 / 4, frame.Height * 3 / 4);
        using var crop = new Mat(frame, roi);
        using var g = new Mat();
        Cv2.CvtColor(crop, g, ColorConversionCodes.BGR2GRAY);
        using var small = new Mat();
        Cv2.Resize(g, small, new Size(), PanScale, PanScale, InterpolationFlags.Area);
        var f = new Mat();
        small.ConvertTo(f, MatType.CV_32F);
        return f;
    }

    long heldSince, lastMatchLog = -10000, lastWideSearch = -10000;
    readonly Stopwatch clock = Stopwatch.StartNew();

    void ShowMap(Mat frame)
    {
        var c = game.Client;
        // Out on the island (or not seen in the Sanctum for a while and the island fits): the
        // island map has door icons too (Sanctum entrances), so the door method must not run.
        if (settings.IslandMap && db.Island is Layout isl && onIsland)
        {
            if (!ShowIslandMap(frame, isl)) Frame?.Invoke(new OverlayFrame(c, false, null, null, null, false, EffectiveDifficulty, ""));
            return;
        }
        if (!settings.IslandMap && outsideStreak >= 3)
        {
            // On the island the map shows Sanctum entrances with the same icons as the doors:
            // don't try to line a Sanctum layout up with it.
            Frame?.Invoke(new OverlayFrame(c, false, null, null, null, false, EffectiveDifficulty, ""));
            return;
        }
        var cands = Candidates();
        // Door icons first: they work under full fog and are fast. Feature matching is the
        // fallback (e.g. a door icon hidden under the cursor), only once the layout is known
        // because it is slow across many candidates.
        // Doors are checked against every layout of the difficulty each time, so a new run is
        // noticed even without passing the party screen; the locked layout is the fallback.
        IReadOnlyList<Layout> all = AllCandidates();
        // The minimap tracking the locked layout right now is independent evidence for it.
        var tracked = CurrentLayout != null && (DateTime.UtcNow - sanctumOkTime).TotalSeconds < 10 ? CurrentLayout : null;
        // Every layout is checked (a difficulty changed just before starting can be read stale);
        // in a tie the difficulty's own layouts win, unless the minimap already found that none
        // of them fits.
        if (settings.ForcedLayout == 0) all = db.Candidates(null);
        var favoured = difficultyDoubted || EffectiveDifficulty == null ? null : db.Candidates(EffectiveDifficulty).Select(l => l.Number).ToHashSet();
        var pose = doors.Match(frame, all, out var arrow, tracked, favoured);
        LastMatchInfo = doors.LastInfo;
        if (pose == null && CurrentLayout != null && all.Count > 1 && doors.LastMapOpen && doors.LastOrange != null && doors.LastBlue != null)
        {
            pose = doors.Match(frame, [CurrentLayout], out arrow);
            if (pose != null) LastMatchInfo += " | kept " + CurrentLayout.Title;
        }
        bool logMatch = false;
        if (clock.ElapsedMilliseconds - lastMatchLog > 1000)
        {
            lastMatchLog = clock.ElapsedMilliseconds;
            logMatch = true;
        }
        // 1. A first guess: both doors > the pan since the last frame > one door at the last zoom
        //    > the last pose. 2. Refine it by matching the explored floor (exact position and
        //    zoom, any zoom level, no doors needed). Feature matching is the last resort.
        var doorPose = pose;
        MapPose? guess = doorPose;
        string src = doorPose != null ? "doors" : "";
        using var panNow = doors.LastMapOpen ? PanImage(frame) : null;
        if (guess == null && panNow != null && prevGray != null && prevGrayPose != null && prevGray.Size() == panNow.Size())
        {
            using var win = new Mat();
            Cv2.CreateHanningWindow(win, panNow.Size(), MatType.CV_32F);
            var shift = Cv2.PhaseCorrelate(prevGray, panNow, win, out double response);
            if (response > 0.08)
            {
                var pp = prevGrayPose;
                guess = new MapPose(pp.Layout, pp.Scale, pp.Tx + shift.X / PanScale, pp.Ty + shift.Y / PanScale, 30, 0);
                src = "pan";
            }
        }
        if (guess == null && doors.LastMapOpen && lastPose != null && (doors.LastOrange != null || doors.LastBlue != null)
            && (CurrentLayout == null || lastPose.Layout.Number == CurrentLayout.Number))
        {
            var L = lastPose.Layout;
            var (seen, img) = doors.LastBlue is Point2d bp ? (bp, L.Blue.Img) : (doors.LastOrange!.Value, L.Orange.Img);
            guess = new MapPose(L, lastPose.Scale, seen.X - lastPose.Scale * img[0], seen.Y - lastPose.Scale * img[1], 30, 0);
            src = "one door";
        }
        if (guess == null && doors.LastMapOpen && lastPose != null) { guess = lastPose; src = "last"; }

        if (guess != null && doors.LastMapOpen)
        {
            bool sure = src == "doors";
            var refined = refiner.Refine(frame, guess, sure ? 0.06 : 0.15, sure ? 40 : 120);
            double need = sure ? 0.5 : 0.45;
            if ((refined == null || refiner.LastScore < need) && !sure && clock.ElapsedMilliseconds - lastWideSearch > 500)
            {
                // Zoomed or panned a lot since the last fit: search much wider (slower, so not every frame).
                lastWideSearch = clock.ElapsedMilliseconds;
                refined = refiner.Refine(frame, guess, 0.45, 320);
            }
            if (refined != null && refiner.LastScore >= need) { pose = refined; LastMatchInfo += $" | {src} + {refiner.LastInfo}"; }
            else if (sure) { pose = doorPose; LastMatchInfo += $" | doors only ({refiner.LastInfo})"; }
            else if (src is "pan" or "one door") { pose = guess; LastMatchInfo += $" | {src} only ({refiner.LastInfo})"; }
            else LastMatchInfo += $" | lost ({refiner.LastInfo})";
        }
        if (pose == null && doors.LastMapOpen && cands.Count == 1)
        {
            pose = matcher.Match(frame, cands, lastPose);
            LastMatchInfo += " | features " + matcher.LastInfo;
        }
        if (pose != null && panNow != null)
        {
            prevGray?.Dispose();
            prevGray = panNow.Clone();
            prevGrayPose = pose;
        }
        if (logMatch)
            Log.Write($"map: {(pose == null ? "no match" : $"{pose.Layout.Title} scale {pose.Scale:0.000}")} | {LastMatchInfo} | candidates {string.Join(",", cands.Select(x => x.Number))}");
        if (pose != null && arrow is Point2d a && pose.Inliers >= 30)
        {
            // The yellow arrow on the full map is the most exact player position there is.
            player = new Point2d((a.X - pose.Tx) / pose.Scale, (a.Y - pose.Ty) / pose.Scale);
            playerTime = DateTime.UtcNow;
            minimap.Seed(player.Value);
        }
        if (pose != null)
        {
            lastPose = pose;
            lastPoseTime = DateTime.UtcNow;
            if (settings.ForcedLayout == 0) Lock(pose.Layout, "map");
        }
        else if (lastPose != null && doors.LastMapOpen && (DateTime.UtcNow - lastPoseTime).TotalSeconds < 0.4)
        {
            pose = lastPose; // brief miss (animation, cursor over the map): keep the last fit
        }
        else if (settings.SaveDebugFrames)
        {
            SaveDebug(frame, "nomatch");
        }

        string status;
        var layout = pose?.Layout ?? CurrentLayout ?? (cands.Count == 1 ? cands[0] : null);
        if (pose != null) status = $"{pose.Layout.Title}  ·  aligned";
        else if (layout != null) status = $"{layout.Title}  ·  open the full map to align";
        else status = EffectiveDifficulty == null
            ? "Open the in-game map. Tip: open the party screen once so I can read the difficulty."
            : $"{EffectiveDifficulty}: open the in-game map so I can find the layout";
        bool fresh = player != null && (DateTime.UtcNow - playerTime).TotalSeconds < 5;
        // Without the map on screen, stay invisible for a moment: the key may just be part of
        // normal play. Only a longer hold shows the hint / unaligned layout panel.
        // Outside a run (open world, hub) holding the key shows nothing unless a map really fits.
        bool visible = pose != null || (InRun && clock.ElapsedMilliseconds - heldSince > 700);
        Frame?.Invoke(new OverlayFrame(c, visible, layout, pose, player, fresh, EffectiveDifficulty, status, Floor: refiner.Floor, Heading: CurrentHeading));
    }

    /// <summary>Full map on the island: feature matching against the island art. Returns false
    /// when the screen isn't the island map (so the Sanctum methods can run).</summary>
    bool ShowIslandMap(Mat frame, Layout isl)
    {
        var c = game.Client;
        doors.Match(frame, [], out var arrow); // map-open check + the player's arrow
        if (!doors.LastMapOpen) { Frame?.Invoke(new OverlayFrame(c, false, isl, null, null, false, EffectiveDifficulty, "")); return true; }
        var pose = matcher.Match(frame, [isl], islandPose);
        LastMatchInfo = "island " + matcher.LastInfo;
        if (clock.ElapsedMilliseconds - lastMatchLog > 1000)
        {
            lastMatchLog = clock.ElapsedMilliseconds;
            Log.Write($"map: {(pose == null ? "no match" : $"island scale {pose.Scale:0.000} inliers {pose.Inliers}")} | {matcher.LastInfo}");
        }
        if (pose == null || (!onIsland && pose.Inliers < 25)) return false;
        islandPose = pose;
        if (arrow is Point2d a)
        {
            islandPlayer = new Point2d((a.X - pose.Tx) / pose.Scale, (a.Y - pose.Ty) / pose.Scale);
            islandTime = DateTime.UtcNow;
            islandTracker.Seed(islandPlayer.Value);
        }
        onIsland = true;
        Frame?.Invoke(new OverlayFrame(c, true, isl, pose, islandPlayer, true, EffectiveDifficulty, "Egg Heist island · aligned"));
        return true;
    }

    DateTime lastDebugSave;
    void SaveDebug(Mat frame, string tag)
    {
        if ((DateTime.UtcNow - lastDebugSave).TotalSeconds < 3) return;
        lastDebugSave = DateTime.UtcNow;
        var dir = Path.Combine(Settings.Dir, "debug");
        Directory.CreateDirectory(dir);
        Cv2.ImWrite(Path.Combine(dir, $"{DateTime.Now:yyyyMMdd_HHmmss}_{tag}.jpg"), frame);
        File.AppendAllText(Path.Combine(dir, "info.txt"), $"{DateTime.Now:HH:mm:ss} {tag} {matcher.LastInfo}{Environment.NewLine}");
    }

    public void Dispose()
    {
        running = false;
        thread?.Join(1000);
    }
}

/// <summary>Count and average/max duration of something since the last report (thread-safe).</summary>
public sealed class PerfStat
{
    readonly object gate = new();
    long n, total, max;

    public void Add(long ticks)
    {
        lock (gate) { n++; total += ticks; if (ticks > max) max = ticks; }
    }

    public string Report()
    {
        lock (gate)
        {
            double ms = 1000.0 / Stopwatch.Frequency;
            string r = n == 0 ? "-" : $"{n / 10.0:0.#}/s avg {total * ms / n:0.0} max {max * ms:0} ms";
            n = total = max = 0;
            return r;
        }
    }
}
