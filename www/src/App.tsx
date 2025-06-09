import wasmURL from "noodles-wasm/noodles_wasm_bg.wasm?url"
import init, { SizedGrid, contour } from 'noodles-wasm';
import { useEffect, useState } from 'react';
import { Box, Button, Editable } from '@chakra-ui/react';

type Point = [number, number];
type Edge = [number, number];

interface Polygon {
  points: Point[],
  edges: Edge[]
}

function App() {

  const [wasmReady, setWasmReady] = useState<boolean>(false);
  const [expr, setExpr] = useState<string>("(x^2 + y^2 - 1.)^3 - x^2 * y^3");
  const [polygon, setPolygon] = useState<Polygon>();

  useEffect(() => {
    init(wasmURL).then(() => setWasmReady(true)).catch(() => setWasmReady(false));
  }, []);

  const onTransform = () => {
    const grid = new SizedGrid(-2.5, -2.5, 5.0, 5.0, 250, 250);
    const res = contour(expr, grid);
    const json = res.serialize();
    const res_poly: Polygon = JSON.parse(json);
    setPolygon(res_poly);
  };

  return (
    <>
      {
        wasmReady ?
          <Box width="100vw" height="100vh" display="flex" justifyContent="center">
            <Box width="100%" height="100%" maxW="1280px">
              <Box position="absolute" width="100%" maxW="1280px" zIndex='1' display="flex">
                <Editable.Root
                  value={expr}
                  onValueChange={(e) => setExpr(e.value)}
                >
                  <Editable.Preview />
                  <Editable.Input />
                </Editable.Root>
                <Button onClick={onTransform}>Segment</Button>
              </Box>
            </Box>
            <Box position="absolute" width="100%" height="100%" display="flex" justifyContent="center">
              <svg width="100%" height="100%" viewBox="-2.5 -2.5 5.0 5.0">
                {
                  //<line x1="-2.5" y1="-2.5" x2="2.5" y2="2.5" stroke="black"/>
                  polygon?.edges.map(([start, end], i) => {
                    const [x1, y1] = polygon.points[start];
                    const [x2, y2] = polygon.points[end];
                    return (
                      <line
                        key={i}
                        x1={x1}
                        y1={-y1}
                        x2={x2}
                        y2={-y2}
                        stroke="black"
                        strokeWidth="0.01"
                      />
                    )
                  })
                }
              </svg>
            </Box>
          </Box> :
          <> Loading WASM </>
      }
    </>
  )
}

export default App
