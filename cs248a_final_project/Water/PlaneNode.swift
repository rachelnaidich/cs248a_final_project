//
//  PlaneNode.swift
//  cs248a_final_project
//
//  Created by Rachel Naidich on 3/21/23.
//

import ARKit
import SceneKit
import MetalKit

private let PLANE_SCALE = Float(0.75)
private let PLANE_SEGS = 60


enum EffectMode {
    case Water
    case Rainbow
}

class PlaneNode: NSObject {
        
    public let contentNode: SCNNode

    private let geometryNode: SCNNode
    
    private let planeMaterial: SCNMaterial

    private let sceneView: ARSCNView
    private let viewportSize: CGSize
    
    private var time: Float = 0.0

    
    init(sceneView: ARSCNView, viewportSize: CGSize, mode: EffectMode) {
        self.sceneView = sceneView
        self.viewportSize = viewportSize
        
        let plane = SCNPlane(width: 1.0, height: 1.0)
        plane.widthSegmentCount = PLANE_SEGS
        plane.heightSegmentCount = PLANE_SEGS
        
        contentNode = SCNNode()
        
        geometryNode = SCNNode(geometry: plane)
        geometryNode.eulerAngles = SCNVector3(-Float.pi / 2, 0, 0)
        geometryNode.scale = SCNVector3(PLANE_SCALE, PLANE_SCALE, PLANE_SCALE)
        contentNode.addChildNode(geometryNode)

        switch mode {
        case .Rainbow:
            planeMaterial = PlaneNode.createMaterial(vertexFunctionName: "geometryEffectVertexShader", fragmentFunctionName: "rainbowFragmentShader")
        case .Water:
            planeMaterial = PlaneNode.createMaterial(vertexFunctionName: "geometryEffectVertexShader", fragmentFunctionName: "waterFragmentShader")
        }
        planeMaterial.setValue(SCNMaterialProperty(contents: sceneView.scene.background.contents!), forKey: "diffuseTexture")
        geometryNode.geometry!.firstMaterial = planeMaterial


        super.init()
    }
    
    func update(time: TimeInterval, timeDelta: Float) {
        self.time += timeDelta
        guard let frame = sceneView.session.currentFrame else { return }
        
        let affineTransform = frame.displayTransform(for: .portrait, viewportSize: viewportSize)
        var transform = SCNMatrix4()
        transform.m11 = Float(affineTransform.a)
        transform.m12 = Float(affineTransform.b)
        transform.m21 = Float(affineTransform.c)
        transform.m22 = Float(affineTransform.d)
        transform.m41 = Float(affineTransform.tx)
        transform.m42 = Float(affineTransform.ty)
        transform.m33 = 1
        transform.m44 = 1
        
        let material = geometryNode.geometry!.firstMaterial!
        material.setValue(SCNMatrix4Invert(transform), forKey: "u_displayTransform")
        material.setValue(NSNumber(value: self.time), forKey: "u_time")
        //geometryNode.position += SCNVector3(0.001, 0, 0)
    }
    
    private static func createMaterial(vertexFunctionName: String, fragmentFunctionName: String)-> SCNMaterial {
        let program = SCNProgram()
        program.vertexFunctionName = vertexFunctionName
        program.fragmentFunctionName = fragmentFunctionName
        
        let material = SCNMaterial()
        material.program = program
        
        return material
    }
}


