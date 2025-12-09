# Get the user's PowerShell profile path
$profilePath = $PROFILE

# Create PowerShell profile if it doesn't exist
if (!(Test-Path -Path $profilePath)) {
    New-Item -ItemType File -Path $profilePath -Force
}

# Add the conda activation function to the profile
$profileContent = @"
# Create a function to activate conda environment
function Activate-Pyrfume {
    C:\ProgramData\Anaconda3\Scripts\activate.bat PyrfumeThesis
}

# Create an alias for the function
Set-Alias -Name pyrfume -Value Activate-Pyrfume

# Export the function so it's available in the current session
Export-ModuleMember -Function Activate-Pyrfume -Alias pyrfume
"@

# Write the content to the profile
Set-Content -Path $profilePath -Value $profileContent

# Force reload the profile
. $profilePath

Write-Host "Conda alias 'pyrfume' has been set up successfully!"
Write-Host "You can now use 'pyrfume' command to activate your environment."
Write-Host "Try it by typing: pyrfume" 