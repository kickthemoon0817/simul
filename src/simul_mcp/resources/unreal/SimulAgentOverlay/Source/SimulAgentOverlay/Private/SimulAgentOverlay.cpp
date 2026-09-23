#include "SimulAgentOverlayLibrary.h"

#include "Containers/Ticker.h"
#include "Editor.h"
#include "ILevelEditor.h"
#include "LevelEditor.h"
#include "SLevelViewport.h"
#include "Dom/JsonObject.h"
#include "Rendering/DrawElements.h"
#include "Serialization/JsonSerializer.h"
#include "Styling/CoreStyle.h"
#include "Widgets/SLeafWidget.h"

namespace
{
constexpr double Lifetime = 120.0;
constexpr int32 MarkerLimit = 64;

struct FMarker
{
    FString AgentId, Activity;
    FVector2D Position;
    FLinearColor Color;
    double Updated;
};

class SAgentOverlay : public SLeafWidget
{
public:
    SLATE_BEGIN_ARGS(SAgentOverlay) {} SLATE_END_ARGS()
    void Construct(const FArguments& Args) { SetVisibility(EVisibility::HitTestInvisible); }
    TMap<FString, FMarker> Markers;

    virtual FVector2D ComputeDesiredSize(float) const override { return FVector2D::ZeroVector; }
    virtual int32 OnPaint(const FPaintArgs& Args, const FGeometry& Geometry,
                         const FSlateRect& CullingRect, FSlateWindowElementList& Elements,
                         int32 Layer, const FWidgetStyle& Style, bool Enabled) const override
    {
        const FVector2D Size = Geometry.GetLocalSize();
        const FSlateFontInfo Font = FCoreStyle::GetDefaultFontStyle("Regular", 12);
        const double Now = FPlatformTime::Seconds();
        TArray<FString> Keys;
        Markers.GetKeys(Keys);
        Keys.Sort();
        Elements.PushClip(FSlateClippingZone(Geometry));
        int32 Row = 0;
        for (const FString& Key : Keys)
        {
            const FMarker& M = Markers[Key];
            if (Now - M.Updated >= Lifetime) { continue; }
            const FVector2D P(M.Position.X * (Size.X - 1), (1 - M.Position.Y) * (Size.Y - 1));
            TArray<FVector2D> Arrow = { P, P + FVector2D(2, 19), P + FVector2D(7, 13),
                                     P + FVector2D(15, 13), P };
            FSlateDrawElement::MakeLines(Elements, Layer, Geometry.ToPaintGeometry(), Arrow,
                                        ESlateDrawEffect::None, M.Color, true, 2.0f);
            auto Text = [&](const FString& Value, const FVector2D& At)
            {
                FSlateDrawElement::MakeText(Elements, Layer + 1,
                    Geometry.ToPaintGeometry(Size, FSlateLayoutTransform(At)), Value,
                    Font, ESlateDrawEffect::None, M.Color);
            };
            Text(M.AgentId, FVector2D(FMath::Clamp(P.X + 18, 0.0, FMath::Max(0.0, Size.X - 180)),
                                     FMath::Clamp(P.Y, 0.0, FMath::Max(0.0, Size.Y - 20))));
            // Separate rows make coincident pointers identifiable.
            Text(M.AgentId + TEXT(": ") + M.Activity, FVector2D(16, 38 + 20 * Row++));
        }
        Elements.PopClip();
        return Layer + 1;
    }
};

struct FViewportOverlay
{
    TWeakPtr<SLevelViewport> Viewport;
    TSharedPtr<SAgentOverlay> Widget;
};
TMap<FString, FViewportOverlay> Overlays;

TSharedPtr<SLevelViewport> FindViewport(const FString& Key)
{
    auto* Module = FModuleManager::GetModulePtr<FLevelEditorModule>("LevelEditor");
    const TSharedPtr<ILevelEditor> Editor = Module ? Module->GetLevelEditorInstance().Pin() : nullptr;
    if (Editor)
    {
        for (const auto& View : Editor->GetViewports())
        {
            if (View && View->GetConfigKey().ToString() == Key) { return View; }
        }
    }
    return nullptr;
}

void RemoveOverlay(FViewportOverlay& Overlay)
{
    if (const auto View = Overlay.Viewport.Pin()) { View->RemoveOverlayWidget(Overlay.Widget.ToSharedRef()); }
}

void Reset()
{
    for (auto& Pair : Overlays) { RemoveOverlay(Pair.Value); }
    Overlays.Empty();
}

bool Prune(float)
{
    const double Now = FPlatformTime::Seconds();
    for (auto It = Overlays.CreateIterator(); It; ++It)
    {
        auto& Overlay = It.Value();
        if (!Overlay.Viewport.IsValid() || FindViewport(It.Key()) != Overlay.Viewport.Pin())
        {
            RemoveOverlay(Overlay);
            It.RemoveCurrent();
            continue;
        }
        auto& Markers = Overlay.Widget->Markers;
        for (auto M = Markers.CreateIterator(); M; ++M)
        {
            if (Now - M.Value().Updated >= Lifetime) { M.RemoveCurrent(); }
        }
        Overlay.Widget->Invalidate(EInvalidateWidgetReason::Paint);
        if (Markers.IsEmpty()) { RemoveOverlay(Overlay); It.RemoveCurrent(); }
    }
    return true;
}

TSharedRef<FJsonObject> ToJson(const FString& Viewport, const FMarker& M)
{
    auto Json = MakeShared<FJsonObject>();
    Json->SetStringField("agent_id", M.AgentId);
    Json->SetStringField("viewport", Viewport);
    Json->SetStringField("label", M.Activity);
    Json->SetStringField("cursor_kind", "agent_overlay");
    Json->SetBoolField("system_cursor_moved", false);
    Json->SetNumberField("expires_in", FMath::Max(0.0, Lifetime - (FPlatformTime::Seconds() - M.Updated)));
    Json->SetArrayField("position", {MakeShared<FJsonValueNumber>(M.Position.X), MakeShared<FJsonValueNumber>(M.Position.Y)});
    Json->SetArrayField("color", {MakeShared<FJsonValueNumber>(M.Color.R), MakeShared<FJsonValueNumber>(M.Color.G),
                                 MakeShared<FJsonValueNumber>(M.Color.B), MakeShared<FJsonValueNumber>(M.Color.A)});
    return Json;
}

FString Encode(const TSharedRef<FJsonObject>& Object)
{
    FString Result;
    FJsonSerializer::Serialize(Object, TJsonWriterFactory<>::Create(&Result));
    return Result;
}

FString OverlayError(const FString& Message)
{
    auto Object = MakeShared<FJsonObject>();
    Object->SetStringField("error", Message);
    return Encode(Object);
}

class FSimulAgentOverlayModule : public IModuleInterface
{
    FTSTicker::FDelegateHandle TickHandle;
    FDelegateHandle MapHandle;
public:
    virtual void StartupModule() override
    {
        TickHandle = FTSTicker::GetCoreTicker().AddTicker(FTickerDelegate::CreateStatic(&Prune), 1.0f);
        MapHandle = FEditorDelegates::MapChange.AddLambda([](uint32 Flags) { if (Flags & 5) { Reset(); } });
    }
    virtual void ShutdownModule() override
    {
        FTSTicker::GetCoreTicker().RemoveTicker(TickHandle);
        FEditorDelegates::MapChange.Remove(MapHandle);
        Reset();
    }
};
}

IMPLEMENT_MODULE(FSimulAgentOverlayModule, SimulAgentOverlay)

FString USimulAgentOverlayLibrary::UpdateCursor(const FString& Viewport, const FString& AgentId,
                                               FVector2D Position, const FString& Activity)
{
    if (!IsInGameThread()) { return OverlayError("Agent overlays require the editor thread"); }
    if (AgentId.TrimStartAndEnd().IsEmpty() || AgentId.Len() > 64 || Activity.Len() > 256 ||
        !FMath::IsFinite(Position.X) || !FMath::IsFinite(Position.Y) ||
        Position.X < 0 || Position.X > 1 || Position.Y < 0 || Position.Y > 1)
    {
        return OverlayError("Invalid agent label, activity, or normalized cursor position");
    }
    Prune(0);
    const auto View = FindViewport(Viewport);
    if (!View) { return OverlayError("Attached level viewport is unavailable"); }
    auto* Overlay = Overlays.Find(Viewport);
    if (!Overlay)
    {
        FViewportOverlay New;
        New.Viewport = View;
        New.Widget = SNew(SAgentOverlay);
        View->AddOverlayWidget(New.Widget.ToSharedRef());
        Overlay = &Overlays.Add(Viewport, New);
    }
    auto& Markers = Overlay->Widget->Markers;
    if (!Markers.Contains(AgentId) && Markers.Num() >= MarkerLimit)
    {
        FString Oldest;
        double Time = TNumericLimits<double>::Max();
        for (const auto& Pair : Markers)
        {
            if (Pair.Value.Updated < Time) { Oldest = Pair.Key; Time = Pair.Value.Updated; }
        }
        Markers.Remove(Oldest);
    }
    const FLinearColor Palette[] = { {0.15f,0.85f,1,1}, {1,0.55f,0.2f,1}, {0.8f,0.4f,1,1}, {0.3f,1,0.5f,1} };
    FMarker Marker { AgentId, Activity, Position, Palette[FCrc::StrCrc32(*AgentId) % 4], FPlatformTime::Seconds() };
    Markers.Add(AgentId, Marker);
    Overlay->Widget->Invalidate(EInvalidateWidgetReason::Paint);
    return Encode(ToJson(Viewport, Marker));
}

FString USimulAgentOverlayLibrary::InspectCursors(const FString& Viewport)
{
    Prune(0);
    auto Object = MakeShared<FJsonObject>();
    TArray<TSharedPtr<FJsonValue>> Markers;
    if (const auto* Overlay = Overlays.Find(Viewport))
    {
        for (const auto& Pair : Overlay->Widget->Markers)
        { Markers.Add(MakeShared<FJsonValueObject>(ToJson(Viewport, Pair.Value))); }
    }
    Object->SetArrayField("agent_cursors", Markers);
    return Encode(Object);
}

bool USimulAgentOverlayLibrary::ClearCursor(const FString& Viewport, const FString& AgentId)
{
    auto* Overlay = Overlays.Find(Viewport);
    const bool Removed = Overlay && Overlay->Widget->Markers.Remove(AgentId) > 0;
    Prune(0);
    return Removed;
}
