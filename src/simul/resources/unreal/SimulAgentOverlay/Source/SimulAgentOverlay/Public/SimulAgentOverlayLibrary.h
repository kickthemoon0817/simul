#pragma once

#include "Kismet/BlueprintFunctionLibrary.h"
#include "SimulAgentOverlayLibrary.generated.h"

/** In-memory, non-interactive overlays in level editor viewports only. */
UCLASS()
class SIMULAGENTOVERLAY_API USimulAgentOverlayLibrary : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()
public:
    UFUNCTION(BlueprintCallable, Category="Simul|Agent")
    static FString UpdateCursor(const FString& Viewport, const FString& AgentId,
                                FVector2D Position, const FString& Activity);

    UFUNCTION(BlueprintCallable, Category="Simul|Agent")
    static FString InspectCursors(const FString& Viewport);

    UFUNCTION(BlueprintCallable, Category="Simul|Agent")
    static bool ClearCursor(const FString& Viewport, const FString& AgentId);
};
