import './App.css'
import wasmURL from "noodles-wasm/noodles_wasm_bg.wasm?url"
import init from 'noodles-wasm';
import { useEffect, useState } from 'react';


function App() {

  const [wasmReady, setWasmReady] = useState<boolean>(false);

  useEffect(() => {
    init(wasmURL).then(() => setWasmReady(true)).catch(() => setWasmReady(false));
  }, []);

  return (
    <>
      {
        wasmReady ? <>Ready</> : <> Loading WASM </>
      }
    </>
  )
}

export default App
