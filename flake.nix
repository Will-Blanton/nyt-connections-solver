{
  description =
    "Python + CUDA Nix flake template for ML/dev environments (venv with uv)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # ─── Centralized version configuration ───
        pythonVersion = "python311"; # e.g. "python312"
        cudaVersion = "cudaPackages_12_8"; # e.g. "cudaPackages_11_8"
        venvDir = ".venv";

        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        python = pkgs.${pythonVersion};
        cuda = pkgs.${cudaVersion};

        pythonPackages = pkgs.${pythonVersion};

        inherit (pkgs.linuxPackages) nvidia_x11;

      in {
        devShells.default = pkgs.mkShell {
          name = "python-cuda-dev";

          buildInputs = [
            python
            pythonPackages.pkgs.venvShellHook
            pkgs.chromium
            pkgs.chromedriver
          ];

          packages = with pkgs; [ jupyter zlib ];

          inherit venvDir;

          # ensure uv installs to the correct location (NOT NIX STORE)
          postVenvCreation = ''
            unset SOURCE_DATE_EPOCH

            uv sync
            uv run python -V
            uv run python -c "import sys; print(sys.executable)"

          '';

          # ensure linker can find required C/C++ libraries and CUDA runtime while always preferring the host driver
          postShellHook = ''
                        unset SOURCE_DATE_EPOCH
                        export CUDA_PATH=${cuda.cudatoolkit.lib}

                        # host driver first, then toolkit libs, then any existing value (open gl points cuda at the global installation, since the shell doesn't fully create its own)
                        LIB_PATHS=(
                          "/run/opengl-driver/lib"
                          "${cuda.cudatoolkit.lib}/lib"
                          "${pkgs.zlib}/lib"
                          ${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}
                        )

                        for path in "''${LIB_PATHS[@]}"; do
                          export LD_LIBRARY_PATH="$path:$LD_LIBRARY_PATH"
                        done

            	          # may not be needed
                        export EXTRA_LDFLAGS="-l/lib -l${nvidia_x11}/lib"
                        export EXTRA_CCFLAGS="-i/usr/include"          

                        uv --version || true
          '';

        };
      });
}
