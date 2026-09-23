using UnrealBuildTool;

public class SimulAgentOverlay : ModuleRules
{
    public SimulAgentOverlay(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine" });
        PrivateDependencyModuleNames.AddRange(new[] { "UnrealEd", "LevelEditor", "Slate", "SlateCore", "Json" });
    }
}
